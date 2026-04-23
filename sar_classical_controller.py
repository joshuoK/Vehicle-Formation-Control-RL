"""
sar_classical_controller.py
============================
Classical APF (Artificial Potential Fields) controller for the SAR Swarm.

This file contains ONLY control logic.  It imports SAREnvironment from
sar_environment.py and drives it using hand-designed APF force laws.

The architecture is intentionally symmetric so a future RL agent can
replace the classical policy with one line:

    Classical:  actions = classical_policy(env)     -> env.step(actions)
    RL agent:   actions = rl_agent.predict(obs)     -> env.step(actions)

Run
---
    python sar_classical_controller.py

Requirements
------------
    pip install numpy matplotlib
    sar_environment.py must be in the same directory.

Theory: Artificial Potential Fields (APF)
-----------------------------------------
APF is a reactive motion planner invented by Oussama Khatib (1985).
The robot is treated as a charged particle in a scalar potential field:

    U(q) = U_att(q) + U_rep(q)

    U_att : attractive well centred on the goal -- pulls robot toward it.
    U_rep : repulsive hill centred on each obstacle -- pushes robot away.

The robot follows the negative gradient of U (steepest descent):

    F(q) = -grad U(q) = F_att + F_rep

In this implementation we additionally add a tangential component F_tang
that rotates F_rep ninety degrees toward the goal.  This breaks the
saddle-point deadlock that occurs when a goal lies directly behind an
obstacle (F_att and F_rep are anti-parallel -- gradient is zero or
oscillates).

Situational awareness
---------------------
Controllers receive env.shared_obs.all() -- the communal obstacle map
built incrementally from all robots' sensor readings.  Early in the
mission this may be empty or partial; controllers degrade gracefully to
pure waypoint pursuit / pure formation tracking until obstacles are found.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sar_environment import (
    SAREnvironment,
    wrap, body_to_world,
    FORM_OFFSET,
    V_MAX, OMEGA_MAX, K_HDG,
    WP_THRESH, R_BODY,
    MAP_W, MAP_H, DT,
    BG,
)


# ===========================================================================
# SECTION 1  |  CONTROLLER CONFIGURATION
# ===========================================================================

# -- Simulation loop --------------------------------------------------------
SIM_TIME  = 300.0   # maximum mission duration [s]
VIZ_EVERY = 4       # render every N steps  (raise to speed up the sim)

# -- Leader APF gains -------------------------------------------------------
# The attraction gain K_ATT_L is intentionally LARGER than K_REP_L so the
# leader always snaps back to the planned sweep lane after dodging.
# The influence radius D0_L is kept smaller than the follower value so the
# leader only detours when an obstacle is genuinely close.
K_ATT_L  = 3.0   # waypoint attraction gain
K_REP_L  = 1.8   # obstacle repulsion gain
K_TANG_L = 0.7   # tangential anti-deadlock nudge gain
D0_L     = 1.0   # obstacle influence radius for the leader [m]

# -- Follower APF gains -----------------------------------------------------
# Followers need stronger repulsion than the leader because they must avoid
# obstacles while simultaneously converging to a moving formation slot.
# K_ATT_F drives them toward the ideal body-frame position; K_REP_F pushes
# them away from obstacles and map walls.
# v9 tuning: wider influence radii + stronger repulsion/tangential gains so
# followers react to obstacles earlier and reliably navigate around them.
K_ATT_F  = 1.8   # formation-target attraction gain
K_REP_F  = 3.5   # obstacle + boundary repulsion gain       (was 2.5)
K_TANG_F = 1.3   # tangential nudge gain                    (was 0.9)
D0_OBS_F = 1.8   # obstacle influence radius for followers  (was 2.0) [m]
D0_BND_F = 2.2   # map-boundary influence distance          (was 2.5) [m]

# Gap-escape mode: when obstacle repulsions strongly cancel each other,
# temporarily suppress obstacle push so the follower can pass through.
GAP_ESCAPE_STEPS = 10
GAP_CANCEL_RATIO = 0.35
GAP_NEAR_FRAC    = 0.75

# Motion shaping for responsiveness + damping
LEADER_MIN_COS_ERR   = 0.35
LEADER_MAX_TURN_FRAC = 0.85
FOLLOWER_MIN_COS_ERR = 0.30
FOLLOWER_MAX_TURN_FRAC = 0.75
FOLLOWER_CMD_ALPHA   = 0.65
FOLLOWER_TARGET_MARGIN = 0.8


# ===========================================================================
# SECTION 2  |  LEADER CONTROLLER
# ===========================================================================

class LeaderCtrl:
    """
    Blended APF controller for the Leader robot.

    Force decomposition
    -------------------
    F_att  : Strong attraction toward the next boustrophedon waypoint.
             Gain K_ATT_L > K_REP_L ensures the sweep path always dominates;
             the leader takes the minimum detour needed to clear an obstacle
             and then immediately resumes its lane.

    F_rep  : Mild repulsion from KNOWN obstacles only (shared obstacle map).
             Early in the mission when no obstacles have been discovered,
             this term is zero and the leader behaves as a pure waypoint
             follower.

    F_tang : The repulsive force rotated ninety degrees (CCW) and then
             flipped if necessary to point toward the goal.  This provides
             a lateral push that slides the robot around an obstacle rather
             than letting it oscillate in the saddle point directly in front
             of a goal-blocking obstacle.

    Waypoint advance logic
    ----------------------
    The waypoint index advances when the robot comes within WP_THRESH of
    the current target.  The guard loop prevents skipping multiple waypoints
    if the robot overshoots several at once (rare but possible at high speed).
    """

    def __init__(self, waypoints: list) -> None:
        self.wpts = waypoints
        self.idx  = 0    # index of the current active waypoint

    def __call__(self, robot, known_obstacles: list) -> tuple:
        """
        Compute (v, omega) velocity commands for one DT timestep.

        Parameters
        ----------
        robot            : Robot instance (the leader)
        known_obstacles  : list of (ox, oy, rad) from SharedObstacleMap.all()
                           Controllers should NEVER receive ground-truth.

        Returns
        -------
        (v, omega) as floats, ready to pass into env.step()
        """
        # -- Advance waypoint when close enough ---------------------------
        for _ in range(len(self.wpts)):   # guard: prevent infinite loop
            tgt = np.array(self.wpts[self.idx])
            if np.linalg.norm(tgt - robot.pos) < WP_THRESH:
                self.idx = (self.idx + 1) % len(self.wpts)
            else:
                break   # current waypoint is still far enough -- stop here

        pos      = robot.pos
        tgt      = np.array(self.wpts[self.idx])
        diff_att = tgt - pos
        d_att    = float(np.linalg.norm(diff_att))

        # -- Attractive force (toward next sweep waypoint) ----------------
        # Normalised so the magnitude is K_ATT_L regardless of distance.
        # This avoids the robot accelerating wildly when far from the target.
        F_att = K_ATT_L * diff_att / max(d_att, 1e-6)

        # -- Repulsive force from all KNOWN obstacles ----------------------
        # Only acts when the robot is within D0_L metres of an obstacle
        # surface (distance to surface = dist_to_centre - obs_radius - R_BODY).
        F_rep = np.zeros(2)
        for ox, oy, rad in known_obstacles:
            diff    = pos - np.array([ox, oy])
            raw     = float(np.linalg.norm(diff))
            d_surf  = max(raw - rad - R_BODY, 0.01)   # distance to surface
            if d_surf < D0_L:
                # Standard APF repulsive gradient:
                #   mag = k * (1/d - 1/d0) / d^2
                mag    = K_REP_L * (1.0 / d_surf - 1.0 / D0_L) / (d_surf ** 2)
                # Direction: unit vector pointing away from obstacle centre
                F_rep += mag * (diff / (raw + 1e-9))

        # -- Tangential nudge (anti-deadlock) -----------------------------
        # Rotates F_rep by 90 degrees CCW to get a perpendicular vector,
        # then flips it toward the goal if needed.  This provides a sideways
        # component that slides the robot around an obstacle rather than
        # pushing it straight back toward the approaching path.
        F_tang     = np.zeros(2)
        f_rep_norm = float(np.linalg.norm(F_rep))
        if f_rep_norm > 1e-6:
            perp = np.array([-F_rep[1], F_rep[0]])   # 90-degree CCW rotation
            if np.dot(perp, diff_att) < 0:
                perp = -perp   # flip so the nudge points toward the goal side
            F_tang = K_TANG_L * perp / (float(np.linalg.norm(perp)) + 1e-9)

        # -- Map total force to unicycle commands --------------------------
        # The robot cannot move sideways, so we extract the DESIRED HEADING
        # from the force vector direction and compute a heading error.
        # v is proportional to cos(err) so the robot slows when turning sharply.
        F   = F_att + F_rep + F_tang
        err = wrap(float(np.arctan2(F[1], F[0])) - robot.theta)
        v_raw = float(np.clip(V_MAX * np.cos(err), 0.0, V_MAX))
        if abs(err) < 1.1:
            v_raw = max(v_raw, LEADER_MIN_COS_ERR * V_MAX)
        w_raw = float(np.clip(K_HDG * err, -LEADER_MAX_TURN_FRAC * OMEGA_MAX,
                              LEADER_MAX_TURN_FRAC * OMEGA_MAX))
        v = v_raw
        w = w_raw
        return v, w

    @property
    def current_idx(self) -> int:
        """Current waypoint index (passed to render() for the target marker)."""
        return self.idx


# ===========================================================================
# SECTION 3  |  FOLLOWER CONTROLLER
# ===========================================================================

class APFFollowerCtrl:
    """
    APF controller that tracks a V-formation slot relative to the leader
    while avoiding known obstacles and map boundaries.

    Formation slot targeting
    ------------------------
    The desired follower position is computed each step as:

        target = leader.pos + R(leader.theta) x body_offset

    where R is the 2-D rotation matrix.  Using the rotation matrix rather
    than global-axis offsets means the V-shape formation rotates correctly
    when the leader makes a 180-degree sweep turn.  Without this, followers
    cross paths and tangle during lane transitions.

    Boundary repulsion
    ------------------
    _boundary() adds a separate repulsive force from each of the four map
    walls, preventing followers from being pushed outside the map by the
    obstacle repulsion term.  The influence distance D0_BND_F is larger
    than D0_OBS_F so walls are felt earlier than obstacle surfaces.

    Tangential nudge
    ----------------
    Same mechanism as the leader: rotate the combined repulsive force
    (obstacles + boundaries) by 90 degrees toward the goal to break
    saddle-point deadlocks.
    """

    def __init__(self, body_offset: np.ndarray) -> None:
        # body_offset is fixed for the lifetime of the controller
        self.offset = np.asarray(body_offset, dtype=float)
        self._gap_escape_steps_left = 0
        self._prev_v = 0.0
        self._prev_w = 0.0

    def formation_target(self, leader_pose: np.ndarray) -> np.ndarray:
        """
        Compute the world-frame ideal position for this follower.

        Transforms the fixed body-frame offset using the leader's current
        heading via body_to_world().
        """
        tgt = body_to_world(self.offset, leader_pose)
        # Keep targets inside map so followers do not chase unreachable off-map points.
        tgt[0] = float(np.clip(tgt[0], FOLLOWER_TARGET_MARGIN, MAP_W - FOLLOWER_TARGET_MARGIN))
        tgt[1] = float(np.clip(tgt[1], FOLLOWER_TARGET_MARGIN, MAP_H - FOLLOWER_TARGET_MARGIN))
        return tgt

    def _attractive(self, pos: np.ndarray,
                    target: np.ndarray) -> np.ndarray:
        """
        Unit-magnitude attraction toward the formation target, scaled by
        K_ATT_F.  Normalised to keep the force bounded at all distances.
        """
        diff = target - pos
        d    = float(np.linalg.norm(diff))
        return K_ATT_F * diff / max(d, 1e-6)

    def _repulsive(self, pos: np.ndarray,
                   known_obstacles: list) -> np.ndarray:
        """
        Sum of repulsive forces from all known obstacles within D0_OBS_F.
        Uses the standard APF gradient formula k*(1/d - 1/d0)/d^2.
        """
        F = np.zeros(2)
        for ox, oy, rad in known_obstacles:
            diff   = pos - np.array([ox, oy])
            raw    = float(np.linalg.norm(diff))
            d_surf = max(raw - rad - R_BODY, 0.01)
            if d_surf < D0_OBS_F:
                mag  = K_REP_F * (1.0 / d_surf - 1.0 / D0_OBS_F) / (d_surf ** 2)
                F   += mag * (diff / (raw + 1e-9))
        return F

    def _should_enter_gap_escape(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        known_obstacles: list,
    ) -> bool:
        """Detect trap between nearby obstacles with opposing repulsive pushes."""
        active_forces: list[np.ndarray] = []
        min_dsurf = 1e9

        for ox, oy, rad in known_obstacles:
            diff = pos - np.array([ox, oy])
            raw = float(np.linalg.norm(diff))
            d_surf = max(raw - rad - R_BODY, 0.01)
            if d_surf >= D0_OBS_F:
                continue

            mag = K_REP_F * (1.0 / d_surf - 1.0 / D0_OBS_F) / (d_surf ** 2)
            f_i = mag * (diff / (raw + 1e-9))
            active_forces.append(f_i)
            min_dsurf = min(min_dsurf, d_surf)

        if len(active_forces) < 2:
            return False

        # Cancellation signature: large individual pushes but small vector sum.
        sum_norm = float(np.linalg.norm(np.sum(active_forces, axis=0)))
        indiv_norm_sum = float(sum(np.linalg.norm(f) for f in active_forces))
        if indiv_norm_sum < 1e-6:
            return False

        cancel_ratio = sum_norm / indiv_norm_sum
        near_enough = min_dsurf < (GAP_NEAR_FRAC * D0_OBS_F)

        # Only trigger when still trying to go toward target.
        to_target = target - pos
        has_target_direction = float(np.linalg.norm(to_target)) > 1e-6
        return near_enough and has_target_direction and (cancel_ratio < GAP_CANCEL_RATIO)

    def _boundary(self, pos: np.ndarray) -> np.ndarray:
        """
        Repulsive force from the four map walls.

        Each wall contributes an outward-pointing force when the robot is
        within D0_BND_F metres of it.  The inward-normal vectors are fixed
        for each wall: right (+x), left (-x), top (+y), bottom (-y)... wait
        actually the normals point INWARD (away from the wall, into the map):
          left wall:   +x direction
          right wall:  -x direction
          bottom wall: +y direction
          top wall:    -y direction
        """
        x, y = pos
        F    = np.zeros(2)
        for d, normal in [
            (x,          np.array([ 1.0,  0.0])),   # distance to left wall
            (MAP_W - x,  np.array([-1.0,  0.0])),   # distance to right wall
            (y,          np.array([ 0.0,  1.0])),   # distance to bottom wall
            (MAP_H - y,  np.array([ 0.0, -1.0])),   # distance to top wall
        ]:
            d = max(float(d), 0.01)
            if d < D0_BND_F:
                mag  = K_REP_F * (1.0 / d - 1.0 / D0_BND_F) / (d ** 2)
                F   += mag * normal
        return F

    def _tangential(self, F_rep: np.ndarray,
                    diff_to_target: np.ndarray) -> np.ndarray:
        """
        Rotate the combined repulsive force 90 degrees CCW and flip if
        needed so it points toward the goal side.  Returns a unit-scaled
        nudge force of magnitude K_TANG_F.

        If F_rep is negligible (no obstacles nearby) this returns zero.
        """
        if float(np.linalg.norm(F_rep)) < 1e-6:
            return np.zeros(2)
        perp = np.array([-F_rep[1], F_rep[0]])   # 90-degree CCW
        if np.dot(perp, diff_to_target) < 0:
            perp = -perp   # flip toward goal side
        return K_TANG_F * perp / (float(np.linalg.norm(perp)) + 1e-9)

    def __call__(self, robot, leader_pose: np.ndarray,
                 known_obstacles: list) -> tuple:
        """
        Compute (v, omega, target_world_pos) for one timestep.

        The target_world_pos is returned so the renderer can draw the
        dashed connector line from the leader to each follower's ideal slot.

        Parameters
        ----------
        robot            : Robot instance (follower-1 or follower-2)
        leader_pose      : (3,) array -- (x, y, theta) of the leader
        known_obstacles  : list from SharedObstacleMap.all()

        Returns
        -------
        (v, omega, target) where target is the world-frame ideal slot position
        """
        tgt    = self.formation_target(leader_pose)   # ideal slot position
        pos    = robot.pos

        F_att  = self._attractive(pos, tgt)
        F_obs  = self._repulsive(pos, known_obstacles)
        F_bnd  = self._boundary(pos)

        if self._gap_escape_steps_left > 0:
            self._gap_escape_steps_left -= 1
            F_rep = F_bnd
            F_tang = np.zeros(2)
        else:
            if self._should_enter_gap_escape(pos, tgt, known_obstacles):
                self._gap_escape_steps_left = GAP_ESCAPE_STEPS
                F_rep = F_bnd
                F_tang = np.zeros(2)
            else:
                F_rep = F_obs + F_bnd
                F_tang = self._tangential(F_rep, tgt - pos)

        F      = F_att + F_rep + F_tang

        # Map net force to unicycle commands
        f_mag   = float(np.linalg.norm(F))
        desired = float(np.arctan2(F[1], F[0])) if f_mag > 1e-6 else robot.theta
        err     = wrap(desired - robot.theta)
        v_raw = float(np.clip(V_MAX * np.cos(err), 0.0, V_MAX))
        if abs(err) < 1.2:
            v_raw = max(v_raw, FOLLOWER_MIN_COS_ERR * V_MAX)
        w_raw = float(np.clip(K_HDG * err, -FOLLOWER_MAX_TURN_FRAC * OMEGA_MAX,
                              FOLLOWER_MAX_TURN_FRAC * OMEGA_MAX))

        # Low-pass command smoothing to reduce follower oscillation.
        a = FOLLOWER_CMD_ALPHA
        v = float(np.clip(a * self._prev_v + (1.0 - a) * v_raw, 0.0, V_MAX))
        w = float(np.clip(a * self._prev_w + (1.0 - a) * w_raw,
                          -OMEGA_MAX, OMEGA_MAX))
        self._prev_v = v
        self._prev_w = w
        return v, w, tgt


# ===========================================================================
# SECTION 4  |  MAIN SIMULATION LOOP
# ===========================================================================

def main() -> None:
    """
    Instantiate the environment, build controllers, run the mission loop.

    Each iteration of the loop:
      1. Queries the shared obstacle map (built by the env's sensing step)
      2. Computes classical APF actions for all three robots
      3. Steps the environment
      4. Renders every VIZ_EVERY steps
      5. Exits if all persons have been found or time runs out
    """
    # -- Initialise environment and controllers ----------------------------
    env  = SAREnvironment(seed=2024)
    obs  = env.reset()   # obs is used by RL agents; classical ignores it

    lctrl  = LeaderCtrl(env.waypoints)
    fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
    fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

    # -- Matplotlib setup --------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG)
    plt.ion()   # interactive mode for live updates

    # Initialise formation target placeholders used by the renderer.
    # They are updated by fctrl1/fctrl2 on the first step.
    f1_tgt = env.robots[1].pos.copy()
    f2_tgt = env.robots[2].pos.copy()

    n_steps = int(SIM_TIME / DT)

    # -- Main loop ---------------------------------------------------------
    for step in range(n_steps):
        leader, f1, f2 = env.robots

        # Query the shared obstacle map.
        # This is what all three controllers see -- NOT the ground truth.
        # Early in the mission this list may be empty or very short.
        known_obs = env.shared_obs.all()

        # Compute classical APF actions
        v_l,  w_l          = lctrl(leader, known_obs)
        v_f1, w_f1, f1_tgt = fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, f2_tgt = fctrl2(f2, leader.pose, known_obs)

        # Step the environment (kinematics, sensing, detection, metrics)
        obs, reward, done, info = env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )

        # Render every VIZ_EVERY steps to reduce matplotlib overhead
        if step % VIZ_EVERY == 0:
            env.render(ax,
                       f1_tgt = f1_tgt,
                       f2_tgt = f2_tgt,
                       wp_idx = lctrl.current_idx)
            ax.set_title(
                "SAR Swarm  [Classical APF]  |  t = {:.1f} s  |"
                "  Obs known {}/{}".format(
                    info['t'],
                    info['obs_discovered'],
                    info['obs_total']),
                color   = '#ccc',
                fontsize = 9.5,
                pad     = 8,
            )
            plt.pause(0.001)

        if done:
            print("\nAll persons found at  t = {:.2f} s".format(info['t']))
            break

    # -- Final frame -------------------------------------------------------
    plt.ioff()
    env.render(ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=lctrl.current_idx)
    ax.set_title('SAR Swarm  [Classical APF]  |  MISSION COMPLETE',
                 color='#00e676', fontsize=11, pad=8)
    plt.tight_layout()

    # -- Terminal report ---------------------------------------------------
    m   = env.get_metrics()
    sep = "=" * 54
    print("\n" + sep)
    print("   SAR SWARM  --  CLASSICAL APF  --  FINAL REPORT")
    print(sep)
    print("   Simulation time      : {:.1f} s".format(m['t']))
    print("   Persons found        : {} / {}".format(m['found'], m['total']))
    print("   Obstacle collisions  : {}".format(m['collisions']))
    print("   Obstacles mapped     : {} / {}".format(
        m['obs_discovered'], m['obs_total']))
    print("   Individual finds     : " +
          "  ".join("R{}({})".format(r.rid, len(r.found_ids))
                    for r in env.robots))
    print("-" * 54)
    print("   FORMATION QUALITY")
    fq = ("Tight"    if m['form_dev_mean'] < 1.0 else
            "Moderate" if m['form_dev_mean'] < 2.5 else
          "Poor")
    print("   Mean deviation       : {:.3f} m  ({})".format(
        m['form_dev_mean'], fq))
    print("   Max  deviation       : {:.3f} m".format(m['form_dev_max']))
    print("-" * 54)
    print("   PATH ADHERENCE  (leader vs planned sweep)")
    pq = ("On-path"  if m['path_dev_mean'] < 0.5 else
          "Moderate" if m['path_dev_mean'] < 1.5 else
          "Off-path")
    print("   Mean path deviation  : {:.3f} m  ({})".format(
        m['path_dev_mean'], pq))
    print("   Max  path deviation  : {:.3f} m".format(m['path_dev_max']))
    print(sep)

    plt.show()


if __name__ == '__main__':
    main()
