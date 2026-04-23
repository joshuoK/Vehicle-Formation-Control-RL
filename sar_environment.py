"""
sar_environment.py
==================
SAR Swarm Environment — map, robots, sensors, shared obstacle map, metrics,
and rendering.  Exposes SAREnvironment with a gym-compatible interface so both
the classical APF controller (sar_classical_controller.py) and future RL
agents can import and use it without modification.

Key design decisions
────────────────────
1.  Obstacle ground-truth is HIDDEN at mission start.
    Robots discover obstacles via their 120° FOV sensor and immediately
    broadcast each new detection to SharedObstacleMap.  Controllers query
    that shared map, not the ground truth — modelling radio communication
    in a real multi-robot team.

2.  Three metrics are tracked inside the environment step:
      ▸ Formation deviation  — ‖actual follower pos − ideal body-frame slot‖
      ▸ Path deviation       — perpendicular distance from leader to nearest
                               planned sweep segment
      ▸ Persons found / Obstacle collisions (existing)

3.  Each robot stores a position trail for visualisation.

Gym interface
─────────────
  env = SAREnvironment(seed=2024)
  obs = env.reset()                                         # → np.float32 array
  obs, reward, done, info = env.step([v_l,w_l, v_f1,w_f1, v_f2,w_f2])
  env.render(ax, f1_tgt=..., f2_tgt=..., wp_idx=...)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 │ CONFIGURATION  (all environment constants live here)
# ─────────────────────────────────────────────────────────────────────────────

# ── Map ───────────────────────────────────────────────────────────────────────
MAP_W         = 20.0
MAP_H         = 20.0

# ── Integration ───────────────────────────────────────────────────────────────
DT            = 0.05        # time-step [s]

# ── Robot kinematics ──────────────────────────────────────────────────────────
V_MAX         = 1.4         # max linear velocity  [m/s]
OMEGA_MAX     = np.pi       # max angular velocity [rad/s]
R_BODY        = 0.30        # robot collision radius [m]

# ── Controller shared params (used by both env bookkeeping and controllers) ───
K_HDG         = 3.5         # heading-error P-gain
WP_THRESH     = 0.40        # waypoint acceptance radius [m]

# ── Boustrophedon sweep ───────────────────────────────────────────────────────
SWEEP_X_MIN   = 1.5
SWEEP_X_MAX   = 18.5
SWEEP_Y0      = 1.5
SWEEP_Y1      = 18.5
SWEEP_STEP    = 2.5         # lane spacing [m]

# ── Formation body-frame offsets (forward = +x, left = +y) ───────────────────
FORM_OFFSET   = {
    1: np.array([-2.5,  1.8]),   # follower-1: behind-left
    2: np.array([-2.5, -1.8]),   # follower-2: behind-right
}

# ── Sensor / FOV ──────────────────────────────────────────────────────────────
FOV_ANG       = np.radians(120)  # full cone angle [rad]
FOV_RANGE     = 5.5              # person detection range [m]

# Sensor heading offsets per robot → three 120° cones = 360° total coverage
#   Leader    :   0°  (straight ahead along travel vector)
#   Follower-1: +120° (forward-left)
#   Follower-2: -120° (forward-right)
SENSOR_OFF    = [0.0, np.radians(120), np.radians(-120)]

# ── Obstacle sensing ──────────────────────────────────────────────────────────
OBS_SENSE_RANGE = 5.0     # robot→obstacle-centre range for detection [m]
OBS_GRID_RES    = 0.5     # deduplication grid cell size [m]

# ── Obstacles (ground truth, hidden until sensed) ─────────────────────────────
N_OBS         = 7
OBS_R_MIN     = 0.35
OBS_R_MAX     = 1.0

# ── Persons (rescue targets) ──────────────────────────────────────────────────
N_PERSONS     = 10

# ── Trail visualisation ───────────────────────────────────────────────────────
TRAIL_LEN     = None        # keep full trail history for persistent visual traces

# ── Coverage grid (for area exploration reward) ───────────────────────────────
COV_GRID_RES  = 1.0         # coverage grid cell size [m]
COV_THRESH    = 1.5         # visit radius for marking cell as explored [m]

# ── Colour palette ────────────────────────────────────────────────────────────
C_ROBOT       = ['#4fc3f7', '#ef5350', '#ffa726']   # leader, f1, f2
C_FOV         = ['#4fc3f7', '#ef5350', '#ffa726']
LABEL         = ['Leader', 'Follower-1', 'Follower-2']
BG            = '#12121f'
GRID_C        = '#1e1e30'


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 │ UTILITY MATH
# ─────────────────────────────────────────────────────────────────────────────

def wrap(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    """2-D rotation matrix R(θ)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_to_world(offset_b: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform body-frame 2-D offset to world coordinates.
    pose = [x, y, theta]
    """
    return pose[:2] + rot2d(pose[2]) @ offset_b


def point_to_segment_dist(p: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray) -> float:
    """
    Minimum distance from point p to line segment a→b.
    Returns the perpendicular distance, or endpoint distance if the
    foot of the perpendicular falls outside the segment.
    """
    ab = b - a
    ap = p - a
    t  = float(np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 │ PATH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_ladder_path() -> list:
    """
    Build ordered (x, y) waypoints for a boustrophedon (back-and-forth)
    coverage sweep across the map.
    """
    wpts, y, go_right = [], SWEEP_Y0, True
    while y <= SWEEP_Y1 + 1e-6:
        x_end = SWEEP_X_MAX if go_right else SWEEP_X_MIN
        wpts.append((x_end, y))
        y_next = y + SWEEP_STEP
        if y_next <= SWEEP_Y1 + 1e-6:
            wpts.append((x_end, y_next))   # vertical transition to next lane
        go_right = not go_right
        y = y_next
    return wpts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 │ ROBOT  (unicycle kinematics + trail history)
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    """
    Non-holonomic unicycle robot.

    State  : (x, y, θ) — pose in world frame
    Inputs : (v, ω)    — linear and angular velocity commands
    Trail  : ring-buffer of last TRAIL_LEN positions for visualisation

    Optimisation note: `pos` and `pose` return pre-allocated arrays that are
    updated in-place in step().  This eliminates ndarray allocation on every
    property access (was called 10+ times per env step across all callers).
    """

    def __init__(self, x: float, y: float, theta: float,
                 rid: int, sensor_off: float):
        self.x          = float(x)
        self.y          = float(y)
        self.theta      = float(theta)
        self.rid        = rid
        self.sensor_off = sensor_off
        self.v          = 0.0
        self.w          = 0.0
        self.found_ids  = set()
        self.trail: list = []

        # Pre-allocated position/pose arrays — updated in-place every step.
        self._pos  = np.array([x, y],          dtype=np.float64)
        self._pose = np.array([x, y, theta],   dtype=np.float64)

    def step(self, v: float, w: float):
        """Integrate unicycle kinematics one DT tick and append to trail."""
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -OMEGA_MAX, OMEGA_MAX))
        self.v      = v
        self.w      = w
        self.x     += v * np.cos(self.theta) * DT
        self.y     += v * np.sin(self.theta) * DT
        self.theta  = wrap(self.theta + w * DT)
        self.x      = float(np.clip(self.x, 0.05, MAP_W - 0.05))
        self.y      = float(np.clip(self.y, 0.05, MAP_H - 0.05))
        # Update cached arrays in-place — no allocation
        self._pos[0]  = self.x
        self._pos[1]  = self.y
        self._pose[0] = self.x
        self._pose[1] = self.y
        self._pose[2] = self.theta
        self.trail.append((self.x, self.y))
        if TRAIL_LEN is not None and len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def pose(self) -> np.ndarray:
        return self._pose


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 │ PERSON  (rescue target)
# ─────────────────────────────────────────────────────────────────────────────

class Person:
    """Static rescue target.  Marked detected when a robot's FOV covers it."""
    _ctr = 0

    def __init__(self, x: float, y: float):
        self.x        = float(x)
        self.y        = float(y)
        self.detected = False
        self.pid      = Person._ctr
        Person._ctr  += 1

    @property
    def pos(self) -> np.ndarray:
        # Persons are static so this is only called in legacy code paths;
        # hot-path detection uses the pre-built _px/_py arrays in SAREnvironment.
        return np.array([self.x, self.y])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 │ SHARED OBSTACLE MAP  (inter-robot communication model)
# ─────────────────────────────────────────────────────────────────────────────

class SharedObstacleMap:
    """
    Communal obstacle knowledge base, shared by the whole swarm.

    How it works
    ─────────────
    1.  At mission start the map is empty — robots know nothing.
    2.  Each robot independently detects obstacles inside its 120° FOV cone
        using sense_obstacles().
    3.  Each new detection is immediately added here, modelling near-instant
        radio broadcast to all teammates.
    4.  All controllers query self.all() instead of the hidden ground truth,
        so early in the mission they may have incomplete obstacle knowledge.

    Deduplication
    ─────────────
    Obstacle centres are snapped to a coarse grid (OBS_GRID_RES = 0.5 m) so
    the same physical obstacle does not accumulate multiple entries from
    different viewing angles or robots.
    """

    def __init__(self):
        self._cells: dict = {}          # grid_key → (ox, oy, rad)
        self._cached_list: list = []    # rebuilt only when new obstacle added
        self._dirty: bool = False

    def _key(self, ox: float, oy: float) -> tuple:
        return (round(ox / OBS_GRID_RES), round(oy / OBS_GRID_RES))

    def update(self, ox: float, oy: float, rad: float) -> bool:
        """Register obstacle.  Returns True if this was a NEW discovery."""
        k = self._key(ox, oy)
        if k not in self._cells:
            self._cells[k] = (ox, oy, rad)
            self._dirty = True
            return True
        return False

    def all(self) -> list:
        """Return known obstacles as a cached list — rebuilt only on new discovery."""
        if self._dirty:
            self._cached_list = list(self._cells.values())
            self._dirty = False
        return self._cached_list

    def __len__(self) -> int:
        return len(self._cells)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 │ SENSING
# ─────────────────────────────────────────────────────────────────────────────

_HALF_FOV  = float(FOV_ANG / 2.0)
_TWO_PI    = float(2.0 * np.pi)
_SENSE_SQ  = float(OBS_SENSE_RANGE * OBS_SENSE_RANGE)  # not used directly but kept for reference
_FOV_R_SQ  = float(FOV_RANGE * FOV_RANGE)


def sense_obstacles(robot: Robot,
                    true_obstacles: list,
                    shared_map: SharedObstacleMap) -> int:
    """
    Detect obstacles inside the robot's active 120° FOV cone and broadcast
    each new one to shared_map.

    Vectorised: distance and angle checks are batched over all obstacles at
    once; the update loop only iterates over the small in-FOV subset.
    """
    if not true_obstacles:
        return 0
    obs        = np.asarray(true_obstacles, dtype=np.float64)  # (N, 3)
    rx, ry     = robot.x, robot.y
    dx         = obs[:, 0] - rx
    dy         = obs[:, 1] - ry
    dist       = np.sqrt(dx * dx + dy * dy)
    in_range   = dist < (OBS_SENSE_RANGE + obs[:, 2])
    if not in_range.any():
        return 0
    sensor_hdg = wrap(robot.theta + robot.sensor_off)
    ri         = np.flatnonzero(in_range)
    angles     = np.arctan2(dy[ri], dx[ri])
    rel        = (angles - sensor_hdg + np.pi) % _TWO_PI - np.pi
    in_fov     = np.abs(rel) <= _HALF_FOV
    new        = 0
    for i in ri[in_fov]:
        ox, oy, rad = true_obstacles[i]
        if shared_map.update(ox, oy, rad):
            new += 1
    return new


def fov_detect_persons(robot: Robot, persons: list,
                       px: np.ndarray, py: np.ndarray) -> int:
    """
    Mark persons inside the robot's FOV cone as detected.

    px, py — pre-built float64 arrays of person x/y positions (set at reset).
    Vectorised: distance + angle computed for all undetected persons at once.
    """
    # Collect indices of undetected persons
    undet = [i for i, p in enumerate(persons) if not p.detected]
    if not undet:
        return 0
    idx    = np.asarray(undet, dtype=np.intp)
    rx, ry = robot.x, robot.y
    dx     = px[idx] - rx
    dy     = py[idx] - ry
    d2     = dx * dx + dy * dy
    in_r   = d2 <= _FOV_R_SQ
    if not in_r.any():
        return 0
    sensor_hdg = wrap(robot.theta + robot.sensor_off)
    ri     = np.flatnonzero(in_r)
    angles = np.arctan2(dy[ri], dx[ri])
    rel    = (angles - sensor_hdg + np.pi) % _TWO_PI - np.pi
    in_fov = np.abs(rel) <= _HALF_FOV
    new    = 0
    for local_i in ri[in_fov]:
        p = persons[undet[local_i]]
        p.detected = True
        robot.found_ids.add(p.pid)
        new += 1
    return new


def in_collision(robot: Robot, obstacles: list) -> bool:
    """True if robot body overlaps any obstacle or touches map boundaries.

    Wall check is first (cheapest); obstacle check is vectorised.
    """
    if (robot.x <= R_BODY or robot.x >= MAP_W - R_BODY or
            robot.y <= R_BODY or robot.y >= MAP_H - R_BODY):
        return True
    if not obstacles:
        return False
    obs = np.asarray(obstacles, dtype=np.float64)
    dx  = robot.x - obs[:, 0]
    dy  = robot.y - obs[:, 1]
    thr = obs[:, 2] + R_BODY
    return bool(np.any(dx * dx + dy * dy < thr * thr))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 │ VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_trail(ax, robot: Robot, color: str):
    """
    Render the robot's position history as a gradient-alpha polyline.
    Older segments are more transparent; recent ones are more opaque.
    """
    trail = robot.trail
    n     = len(trail)
    if n < 2:
        return
    xs = [pt[0] for pt in trail]
    ys = [pt[1] for pt in trail]
    for i in range(n - 1):
        alpha = 0.05 + 0.45 * (i / n)
        ax.plot(xs[i:i+2], ys[i:i+2], '-',
                color=color, alpha=alpha, linewidth=1.3, zorder=3,
                solid_capstyle='round')


def draw_fov(ax, robot: Robot, color: str):
    """Draw the robot's active 120° FOV cone as a lightly shaded wedge."""
    hdg_deg = np.degrees(wrap(robot.theta + robot.sensor_off))
    fov_deg = np.degrees(FOV_ANG)
    ax.add_patch(Wedge(
        (robot.x, robot.y), r=FOV_RANGE,
        theta1=hdg_deg - fov_deg / 2,
        theta2=hdg_deg + fov_deg / 2,
        alpha=0.12, facecolor=color, edgecolor=color,
        linewidth=0.6, zorder=2,
    ))


def draw_robot(ax, robot: Robot, color: str, label: str):
    """Plot robot as a dot with a heading arrow and a label."""
    ax.plot(robot.x, robot.y, 'o', color=color,
            markersize=9, zorder=6,
            markeredgecolor='white', markeredgewidth=0.5)
    alen = 0.7
    ax.annotate('',
        xy     = (robot.x + alen * np.cos(robot.theta),
                  robot.y + alen * np.sin(robot.theta)),
        xytext = (robot.x, robot.y),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8),
        zorder=7,
    )
    ax.text(robot.x + 0.15, robot.y + 0.35, label,
            fontsize=7.5, color=color, fontweight='bold', zorder=8)


def draw_scoreboard(ax, metrics: dict):
    """
    HUD panel in the top-left corner.

    The border colour signals formation quality at a glance:
      ● Green  — mean deviation < 1.0 m  (tight formation)
      ● Amber  — mean deviation < 2.5 m  (some drift)
      ● Red    — mean deviation ≥ 2.5 m  (formation breakdown)
    """
    fd = metrics['form_dev_mean']
    border_col = '#00e676' if fd < 1.0 else '#ffa726' if fd < 2.5 else '#ef5350'
    lines = [
        f"  ✦  Persons Found      : {metrics['found']:2d} / {metrics['total']}",
        f"  ⚠  Collisions          : {metrics['collisions']}",
        f"  ◈  Form. Dev  (avg)   : {metrics['form_dev_mean']:.2f} m",
        f"  ◈  Form. Dev  (max)   : {metrics['form_dev_max']:.2f} m",
        f"  ▸  Path  Dev  (avg)   : {metrics['path_dev_mean']:.2f} m",
        f"  ▸  Path  Dev  (max)   : {metrics['path_dev_max']:.2f} m",
        f"  ◉  Obstacles Known    : {metrics['obs_discovered']:2d} / {metrics['obs_total']}",
    ]
    ax.text(0.015, 0.975, "\n".join(lines),
            transform         = ax.transAxes,
            fontsize          = 8.5,
            verticalalignment = 'top',
            color             = 'white',
            fontfamily        = 'monospace',
            bbox              = dict(boxstyle  = 'round,pad=0.45',
                                     facecolor = '#1e1e35',
                                     edgecolor = border_col,
                                     alpha     = 0.88))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 │ SAR ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class SAREnvironment:
    """
    Gym-compatible SAR Swarm environment.

    Observation vector  (normalised float32, length = OBS_DIM)
    ────────────────────────────────────────────────────────────
      5 values per robot  : x/W, y/H, θ/π, v/V_MAX, ω/Ω_MAX
      3 values per obstacle slot (N_OBS) : ox/W, oy/H, rad   — zero-padded
      3 values per person slot  (N_PERSONS): px/W, py/H, detected — zero-padded

    Action vector  [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
    ─────────────────────────────────────────────────────
      Each value is clipped inside Robot.step() to [−V_MAX, V_MAX] and
      [−Ω_MAX, Ω_MAX] respectively.

    Reward  (for RL training)
    ─────────────────────────
      +10  per newly detected person
      −0.5 × formation deviation [m]
      −0.3 × path deviation [m]
      −5   per collision event
    """

    OBS_DIM: int = 15 + 3 * N_OBS + 3 * N_PERSONS

    def __init__(self, seed: int = 202):
        self.seed      = seed
        self._rng      = np.random.default_rng(seed=self.seed)
        self._rng_seed = self.seed
        self.waypoints  = generate_ladder_path()
        wpts_arr        = np.array(self.waypoints, dtype=np.float64)  # (N, 2)
        self.path_segs  = [(wpts_arr[i], wpts_arr[i + 1])
                           for i in range(len(wpts_arr) - 1)]
        # Pre-stacked segment endpoints for vectorised path-deviation query
        self._seg_a = wpts_arr[:-1]          # (S, 2)  segment starts
        self._seg_b = wpts_arr[1:]           # (S, 2)  segment ends
        self._seg_ab   = self._seg_b - self._seg_a                  # (S, 2)
        self._seg_ab2  = np.einsum('ij,ij->i', self._seg_ab, self._seg_ab) + 1e-9  # (S,)
        self._n_active_obstacles = N_OBS  # curriculum: limit obstacles per episode
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """
        Randomise a new episode.  Returns the initial observation vector.
        The shared obstacle map is cleared — robots start with zero knowledge.
        """
        if self._rng_seed != self.seed:
            self._rng = np.random.default_rng(seed=self.seed)
            self._rng_seed = self.seed
        rng         = self._rng
        Person._ctr = 0

        # Ground-truth obstacles
        self.true_obstacles: list = []
        for _ in range(50_000):
            if len(self.true_obstacles) >= self._n_active_obstacles:
                break
            ox  = rng.uniform(3.0, MAP_W - 3.0)
            oy  = rng.uniform(3.0, MAP_H - 3.0)
            rad = rng.uniform(OBS_R_MIN, OBS_R_MAX)
            if np.hypot(ox - 2.0, oy - 1.5) < 3.5:        # protect start area
                continue
            if any(np.hypot(ox - o[0], oy - o[1]) < rad + o[2] + 0.5
                   for o in self.true_obstacles):           # no overlap
                continue
            self.true_obstacles.append((ox, oy, rad))

        # Persons
        self.persons: list = []
        for _ in range(50_000):
            if len(self.persons) >= N_PERSONS:
                break
            px = rng.uniform(1.0, MAP_W - 1.0)
            py = rng.uniform(1.0, MAP_H - 1.0)
            if any(np.hypot(px - o[0], py - o[1]) < o[2] + 0.6
                   for o in self.true_obstacles):
                continue
            self.persons.append(Person(px, py))

        # Robots (initial positions in the lower-left start zone)
        self.robots: list = [
            Robot(2.0, 1.5, 0.0, 0, SENSOR_OFF[0]),   # leader
            Robot(0.5, 3.0, 0.0, 1, SENSOR_OFF[1]),   # follower-1
            Robot(0.5, 0.5, 0.0, 2, SENSOR_OFF[2]),   # follower-2
        ]

        # Pre-built person coordinate arrays for vectorised FOV detection
        self._px = np.array([p.x for p in self.persons], dtype=np.float64)
        self._py = np.array([p.y for p in self.persons], dtype=np.float64)

        # Shared map starts empty — robots must discover obstacles
        self.shared_obs = SharedObstacleMap()

        # ── Metric accumulators ───────────────────────────────────────────
        self.t                = 0.0
        self.step_count       = 0
        self.total_found      = 0
        self.total_collisions = 0
        self._coll_cd         = [0, 0, 0]   # per-robot collision cooldown

        self._form_sum        = 0.0
        self._form_max        = 0.0
        self._path_sum        = 0.0
        self._path_max        = 0.0
        self._n_samples       = 0

        # ── Coverage grid initialization ────────────────────────────────────
        grid_cols = int(np.ceil(MAP_W / COV_GRID_RES))
        grid_rows = int(np.ceil(MAP_H / COV_GRID_RES))
        self.coverage_grid   = np.zeros((grid_rows, grid_cols), dtype=bool)
        self._coverage_count = 0

        # Pre-allocated flat observation buffer — zeroed and reused every step
        self._obs_buf = np.zeros(self.OBS_DIM, dtype=np.float32)

        return self._build_obs()

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions) -> tuple:
        """
        Apply velocity commands and advance one DT tick.

        Parameters
        ----------
        actions : array-like, length 6
            [v_leader, w_leader, v_follower1, w_follower1,
             v_follower2, w_follower2]

        Returns
        -------
        obs    : np.ndarray  — next observation
        reward : float       — step reward signal
        done   : bool        — True when all persons are found
        info   : dict        — full metrics snapshot
        """
        v_l, w_l, v_f1, w_f1, v_f2, w_f2 = actions
        leader, f1, f2 = self.robots

        # 1 ── Kinematics ─────────────────────────────────────────────────
        leader.step(v_l,  w_l)
        f1.step(v_f1, w_f1)
        f2.step(v_f2, w_f2)

        # 2 ── Obstacle sensing + inter-robot sharing ──────────────────────
        for robot in self.robots:
            sense_obstacles(robot, self.true_obstacles, self.shared_obs)

        # 3 ── Person detection ────────────────────────────────────────────
        prev_found = self.total_found
        for robot in self.robots:
            self.total_found += fov_detect_persons(robot, self.persons, self._px, self._py)
        new_detections = self.total_found - prev_found

        # 4 ── Collision bookkeeping ───────────────────────────────────────
        for i, robot in enumerate(self.robots):
            if self._coll_cd[i] > 0:
                self._coll_cd[i] -= 1
            elif in_collision(robot, self.true_obstacles):
                self.total_collisions += 1
                self._coll_cd[i]       = 30   # ~1.5 s cooldown

        # 5 ── Formation deviation ─────────────────────────────────────────
        #      Measure how far each follower is from its ideal body-frame slot
        f1_ideal   = body_to_world(FORM_OFFSET[1], leader.pose)
        f2_ideal   = body_to_world(FORM_OFFSET[2], leader.pose)
        step_fdev  = (float(np.linalg.norm(f1.pos - f1_ideal)) +
                      float(np.linalg.norm(f2.pos - f2_ideal))) / 2.0
        self._form_sum += step_fdev
        self._form_max  = max(self._form_max, step_fdev)

        # 6 ── Path deviation ──────────────────────────────────────────────
        #      Perpendicular distance from leader to nearest planned segment
        _ap  = leader.pos - self._seg_a                          # (S, 2)
        _t   = np.clip(
            np.einsum('ij,ij->i', _ap, self._seg_ab) / self._seg_ab2,
            0.0, 1.0)                                            # (S,)
        _foot = self._seg_a + _t[:, None] * self._seg_ab        # (S, 2)
        _d    = _foot - leader.pos                               # (S, 2)
        step_pdev = float(np.sqrt(np.einsum('ij,ij->i', _d, _d).min()))
        self._path_sum += step_pdev
        self._path_max  = max(self._path_max, step_pdev)

        # 7 ── Coverage grid update ────────────────────────────────────────────
        #      Mark cells as explored near each robot position
        new_cells_explored = 0
        for robot in self.robots:
            new_cells_explored += self._update_coverage(robot)
        self._coverage_count += new_cells_explored

        self._n_samples += 1
        self.t          += DT
        self.step_count += 1

        done   = (self.total_found >= len(self.persons))
        obs    = self._build_obs()
        info   = self.get_metrics()
        reward = self._reward(new_detections, step_fdev, step_pdev, new_cells_explored)

        return obs, reward, done, info

    # ── Observation builder ───────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        """
        Assemble the normalised flat observation vector into the pre-allocated
        buffer.  Returns a VIEW of that buffer — callers must not store it
        across steps (SB3 VecEnv copies it automatically).
        """
        obs = self._obs_buf
        obs[:] = 0.0
        idx = 0

        for r in self.robots:
            obs[idx]     = r.x / MAP_W
            obs[idx + 1] = r.y / MAP_H
            obs[idx + 2] = r.theta / np.pi
            obs[idx + 3] = r.v / V_MAX
            obs[idx + 4] = r.w / OMEGA_MAX
            idx += 5

        known = self.shared_obs.all()
        if known:
            obs_block = np.asarray(known[:N_OBS], dtype=np.float32)
            n_known = obs_block.shape[0]
            obs_block[:, 0] /= MAP_W
            obs_block[:, 1] /= MAP_H
            obs[idx:idx + 3 * n_known] = obs_block.reshape(-1)
        idx += 3 * N_OBS

        if self.persons:
            person_block = np.asarray(
                [(p.x / MAP_W, p.y / MAP_H, float(p.detected))
                 for p in self.persons[:N_PERSONS]],
                dtype=np.float32,
            )
            n_persons = person_block.shape[0]
            obs[idx:idx + 3 * n_persons] = person_block.reshape(-1)

        return obs

    # ── Coverage grid update ─────────────────────────────────────────────────
    def _update_coverage(self, robot) -> int:
        """
        Mark cells in the coverage grid as explored within COV_THRESH of robot.
        Vectorised: slices the 3×3 neighbourhood directly into the grid.
        """
        rows, cols = self.coverage_grid.shape
        cx = int(robot.x / COV_GRID_RES)
        cy = int(robot.y / COV_GRID_RES)
        r0 = max(cy - 1, 0);  r1 = min(cy + 2, rows)
        c0 = max(cx - 1, 0);  c1 = min(cx + 2, cols)
        patch = self.coverage_grid[r0:r1, c0:c1]
        new_cells = int(np.count_nonzero(~patch))
        patch[:] = True
        return new_cells

    # ── Reward ────────────────────────────────────────────────────────────────
    def _reward(self, new_detections: int,
                form_dev: float, path_dev: float, new_cells: int = 0) -> float:
        """
        Shaped reward for RL training.
        Positive signal for finding people and exploring new areas;
        negative for drift and collisions.
        """
        r  =  new_detections         * 10.0
        r +=  new_cells              *  0.15  # reward for exploring new cells
        r -=  form_dev               *  0.5
        r -=  path_dev               *  0.3
        r -=  self.total_collisions  *  5.0
        return float(r)

    # ── Metrics snapshot ──────────────────────────────────────────────────────
    def get_metrics(self) -> dict:
        """Return a dict of all current performance metrics."""
        n = max(self._n_samples, 1)
        return {
            'found':          self.total_found,
            'total':          len(self.persons),
            'collisions':     self.total_collisions,
            'form_dev_mean':  self._form_sum / n,
            'form_dev_max':   self._form_max,
            'path_dev_mean':  self._path_sum / n,
            'path_dev_max':   self._path_max,
            'obs_discovered': len(self.shared_obs),
            'obs_total':      len(self.true_obstacles),
            't':              self.t,
        }

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, ax,
               f1_tgt: np.ndarray = None,
               f2_tgt: np.ndarray = None,
               wp_idx:  int       = 0):
        """
        Draw the full simulation scene onto *ax*.

        Parameters
        ----------
        f1_tgt, f2_tgt : world-frame formation target positions (optional).
                         If provided, dashed lines and × markers are drawn.
        wp_idx         : current leader waypoint index for the target marker.

        Obstacle rendering
        ──────────────────
        Hidden (not yet sensed) obstacles are drawn very dimly.
        Known (already broadcast) obstacles are drawn brightly with a + marker.
        This makes the discovery progress visible at a glance.
        """
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlim(-0.5, MAP_W + 0.5)
        ax.set_ylim(-0.5, MAP_H + 0.5)
        ax.set_aspect('equal')
        ax.tick_params(colors='#888')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

        # Grid lines
        for v in np.arange(0, MAP_W + 1, 5):
            ax.axvline(v, color=GRID_C, lw=0.5)
        for h in np.arange(0, MAP_H + 1, 5):
            ax.axhline(h, color=GRID_C, lw=0.5)

        # Map border
        ax.add_patch(mpatches.Rectangle(
            (0, 0), MAP_W, MAP_H, lw=1.5,
            edgecolor='#556', facecolor='none', zorder=1))

        # Planned sweep path (faint dashed)
        wx = [w[0] for w in self.waypoints]
        wy = [w[1] for w in self.waypoints]
        ax.plot(wx, wy, '--', color='#333355', lw=0.9, zorder=1)

        # Obstacles — dim if hidden, bright if discovered
        known_keys = set(self.shared_obs._cells.keys())
        for ox, oy, rad in self.true_obstacles:
            k       = self.shared_obs._key(ox, oy)
            is_known = k in known_keys
            face    = '#3a3a5c' if is_known else '#1e1e35'
            alpha   = 0.90     if is_known else 0.40
            ax.add_patch(plt.Circle((ox, oy), rad,
                                    color=face, alpha=alpha, zorder=2))
            if is_known:
                ax.plot(ox, oy, '+', color='#aaaaaa',
                        markersize=6, lw=1, zorder=3)

        # Persons
        for p in self.persons:
            if p.detected:
                ax.plot(p.x, p.y, '*', color='#00e676',
                        markersize=13, zorder=9)
                ax.plot(p.x, p.y, 'o', color='none', markersize=16,
                        markeredgecolor='#00e676',
                        markeredgewidth=1.5, zorder=9)
            else:
                ax.plot(p.x, p.y, '+', color='#ffee58',
                        markersize=11, markeredgewidth=2.0, zorder=8)

        # Current waypoint target marker
        wt = np.array(self.waypoints[wp_idx])
        ax.plot(wt[0], wt[1], 's', color='#4fc3f7',
                markersize=7, alpha=0.5, zorder=4)

        # Formation target dashes (leader → follower slots)
        leader = self.robots[0]
        for tgt, col in [(f1_tgt, C_ROBOT[1]), (f2_tgt, C_ROBOT[2])]:
            if tgt is not None:
                ax.plot([leader.x, tgt[0]], [leader.y, tgt[1]],
                        '--', color=col, alpha=0.25, lw=0.8, zorder=3)
                ax.plot(tgt[0], tgt[1], 'x', color=col,
                        markersize=6, alpha=0.5, zorder=4)

        # Trail → FOV cone → Robot  (each layer in increasing z-order)
        for robot, color in zip(self.robots, C_ROBOT):
            draw_trail(ax, robot, color)
        for robot, color in zip(self.robots, C_FOV):
            draw_fov(ax, robot, color)
        for robot, color, lbl in zip(self.robots, C_ROBOT, LABEL):
            draw_robot(ax, robot, color, lbl)

        # HUD
        draw_scoreboard(ax, self.get_metrics())

        # Legend
        legend_elems = [
            plt.Line2D([0], [0], marker='o', linestyle='None',
                        markerfacecolor=C_ROBOT[i], markeredgecolor='white',
                        markersize=8, label=LABEL[i])
            for i in range(3)
        ] + [
            plt.Line2D([0], [0], marker='+', linestyle='None',
                        color='#ffee58', markersize=10,
                        markeredgewidth=2, label='Person (undetected)'),
            plt.Line2D([0], [0], marker='*', linestyle='None',
                        color='#00e676', markersize=10,
                        label='Person (found)'),
            mpatches.Patch(facecolor='#3a3a5c', label='Obstacle (known)'),
            mpatches.Patch(facecolor='#1e1e35', alpha=0.7,
                           label='Obstacle (hidden)'),
        ]
        ax.legend(handles=legend_elems, loc='upper right', fontsize=7.5,
                  facecolor='#1e1e35', edgecolor='#4fc3f7',
                  labelcolor='white')
