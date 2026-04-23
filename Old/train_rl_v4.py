"""
train_rl.py (v4 — Safety Limits & Sensor Coverage)
====================================================
Structural improvements over v3 that do NOT change the reward system:

1.  SENSOR-AWARE COVERAGE MAP
    Coverage is now updated via the robots' actual FOV cone (FOV_ANG, FOV_RANGE)
    rather than just flagging the cell the leader occupies.  All three robots
    contribute to coverage so the map fills faster and more accurately.

2.  SAFETY SPEED LIMITS
    The leader is automatically slowed when it gets close to known obstacles or
    walls (SAFETY_SLOW_CLEARANCE → 50% speed, SAFETY_STOP_CLEARANCE → 15% speed).
    This prevents the RL policy from crashing while still exploring.

3.  PROXIMITY PENALTIES
    Soft exponential penalties are added to the reward when the leader is near
    obstacles or walls, giving a smooth gradient before the hard collision event.

4.  COMMAND SMOOTHING
    An exponential moving average (alpha = COMMAND_SMOOTH_ALPHA) is applied to
    the leader's (v, omega) commands to suppress high-frequency jitter.

5.  DEVICE SUPPORT
    train() and evaluate() accept a --device argument (auto / cpu / cuda).

6.  INCREASED N_STEPS
    N_STEPS raised from 512 → 2048 for better long-horizon credit assignment
    with the LSTM.  (Note: BATCH_SIZE still from v3; fixed in v5.)

All v3 improvements retained: hybrid APF followers, LSTM policy, curriculum,
visited-cell coverage, partial observability fix, seed diversity.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import platform
import multiprocessing
from typing import Optional

import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:
    raise ImportError(
        "sb3-contrib is required for the LSTM policy.\n"
        "Install with:  pip install sb3-contrib"
    ) from exc

sys.path.insert(0, os.path.dirname(__file__))
from sar_environment import (  # noqa: E402
    SAREnvironment,
    V_MAX,
    OMEGA_MAX,
    R_BODY,
    N_PERSONS,
    N_OBS,
    FORM_OFFSET,
    MAP_W,
    MAP_H,
    FOV_ANG,
    FOV_RANGE,
)
from sar_classical_controller import APFFollowerCtrl  # noqa: E402

# ---------------------------------------------------------------------------
# Cross-platform multiprocessing
# ---------------------------------------------------------------------------
multiprocessing.freeze_support()
_IS_WINDOWS = platform.system() == "Windows"
_MP_START_METHOD = "spawn" if _IS_WINDOWS else "fork"


# ===========================================================================
# SECTION 1  |  HYPERPARAMETERS & FILE PATHS
# ===========================================================================

# -- Environment -------------------------------------------------------------
N_ENVS            = 4
MAX_EPISODE_STEPS = 4000
SEED              = 2024

# -- Coverage grid -----------------------------------------------------------
GRID_N       = 10
CELL_SIZE_X  = MAP_W / GRID_N
CELL_SIZE_Y  = MAP_H / GRID_N
COVERAGE_DIM = GRID_N * GRID_N

# -- Curriculum  [(start_timestep, n_active_persons)] -----------------------
CURRICULUM_STAGES = [
    (        0,  3),
    (  500_000,  5),
    (1_200_000,  7),
    (2_500_000, 10),
]

# -- Reward shaping (v4 — proximity penalties added; other values from v3) --
R_PERSON_FOUND    =  50.0   # per newly detected rescue target
R_COLLISION       = -40.0   # per new collision (no episode termination)
R_FORWARD         =   0.0   # dead code — kept for compatibility, always 0
R_NO_COLLISION_STEP =  0.0  # dead code — kept for compatibility, always 0
R_FORMATION_BONUS =   1.0   # when form_dev < 0.5 m
R_COVERAGE        =   0.5   # first visit to a new grid cell (sensor-based)
R_TIME_PENALTY    =  -0.02  # per step
R_NEAR_OBS_EXP_MAX  = -5.0  # max exponential penalty near obstacles
R_NEAR_WALL_EXP_MAX = -5.0  # max exponential penalty near walls
EXP_OBS_ALPHA     =   4.0
EXP_WALL_ALPHA    =   5.0

# -- Safety and control blend -----------------------------------------------
SAFETY_SLOW_CLEARANCE = 1.5   # below this distance → clamp to 50% V_MAX
SAFETY_STOP_CLEARANCE = 0.9   # below this distance → clamp to 15% V_MAX
WALL_SAFE_CLEARANCE   = 1.2
COMMAND_SMOOTH_ALPHA  = 0.6   # EMA smoothing on (v, omega) commands

# -- LSTM / RecurrentPPO hyperparameters ------------------------------------
LEARNING_RATE  = 3e-4
N_STEPS        = 2048   # ↑ from 512 (v3); better long-horizon credit assignment
BATCH_SIZE     = 128    # NOTE: does not divide N_STEPS cleanly for RecurrentPPO
                        #       (fixed in v5 to 512)
N_EPOCHS       = 10
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.01
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5

POLICY_KWARGS = dict(
    lstm_hidden_size   = 256,
    n_lstm_layers      = 1,
    shared_lstm        = False,
    enable_critic_lstm = True,
    net_arch           = dict(pi=[128, 128], vf=[128, 128]),
)

# -- Training schedule -------------------------------------------------------
TOTAL_TIMESTEPS  = 5_000_000
CHECKPOINT_FREQ  = 100_000
EVAL_FREQ        = 50_000
EVAL_EPISODES    = 5
N_EVAL_EPISODES  = 3

# -- File paths --------------------------------------------------------------
MODEL_PATH      = "ppo_swarm_agent"
TENSORBOARD_LOG = "./ppo_sar_tensorboard/"
CHECKPOINT_DIR  = "./checkpoints/"


# ===========================================================================
# SECTION 2  |  GYMNASIUM WRAPPER
# ===========================================================================

class SARGymnasiumWrapper(gym.Env):
    """
    Gymnasium wrapper with hybrid APF+RL control, sensor-based coverage map,
    safety speed limits, and curriculum learning.

    Observation layout  (float32, length = SAREnvironment.OBS_DIM + COVERAGE_DIM)
    ------------------------------------------------------------------------------
      [0:15]    Robot poses/velocities  (5 values x 3 robots)
      [15:36]   Known obstacles         (3 values x N_OBS slots, zero-padded)
      [36:66]   Person slots            (3 values x N_PERSONS)
                  px/W and py/H are ZEROED for undetected persons.
      [66:166]  Coverage grid           (100 binary values: 0=unvisited, 1=visited)

    Action space  (2-D, normalised [-1, 1])
      [0]  v_leader  -> [0, V_MAX]
      [1]  w_leader  -> [-OMEGA_MAX, OMEGA_MAX]
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM  = SAREnvironment.OBS_DIM + COVERAGE_DIM

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed        = seed
        self._episode_count    = 0
        self._n_active_persons = CURRICULUM_STAGES[0][1]

        self._env    = SAREnvironment(seed=seed)
        self._fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
        self._fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._step_count:       int        = 0
        self._prev_found:       int        = 0
        self._prev_collisions:  int        = 0
        self._prev_v_l:         float      = 0.0
        self._prev_w_l:         float      = 0.0
        self._coverage_grid:    np.ndarray = np.zeros(COVERAGE_DIM, dtype=np.float32)

        # Pre-compute cell centers for sensor coverage checks
        x_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_X
        y_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_Y
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        self._coverage_cell_centers = np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)

    # ── Curriculum ────────────────────────────────────────────────────────────

    def set_n_active_persons(self, n: int) -> None:
        """Update active persons count.  Called by CurriculumCallback."""
        self._n_active_persons = int(np.clip(n, 1, N_PERSONS))

    def _apply_curriculum(self) -> None:
        """Pre-detect excess persons so only _n_active_persons are real targets."""
        pre_found = 0
        for i, p in enumerate(self._env.persons):
            if i >= self._n_active_persons and not p.detected:
                p.detected = True
                pre_found += 1
        self._env.total_found += pre_found

    # ── Coverage (sensor-based) ───────────────────────────────────────────────

    @staticmethod
    def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    def _update_sensor_coverage(self) -> int:
        """
        Mark cells as visited if they fall within any robot's FOV cone.
        Returns the count of newly covered cells.
        """
        centers  = self._coverage_cell_centers
        observed = np.zeros(COVERAGE_DIM, dtype=bool)
        half_fov = 0.5 * FOV_ANG

        for robot in self._env.robots:
            dx = centers[:, 0] - float(robot.x)
            dy = centers[:, 1] - float(robot.y)
            d2 = dx * dx + dy * dy
            in_range = d2 <= (FOV_RANGE * FOV_RANGE)
            if not np.any(in_range):
                continue

            sensor_hdg = float(robot.theta + robot.sensor_off)
            angles     = np.arctan2(dy, dx)
            rel        = self._wrap_to_pi(angles - sensor_hdg)
            in_cone    = in_range & (np.abs(rel) <= half_fov)
            observed  |= in_cone

        newly_observed = observed & (self._coverage_grid < 0.5)
        new_count = int(np.count_nonzero(newly_observed))
        if new_count > 0:
            self._coverage_grid[newly_observed] = 1.0
        return new_count

    # ── Observation ───────────────────────────────────────────────────────────

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Apply partial-observability mask then append coverage grid."""
        obs = raw_obs.copy()
        person_start = 15 + 3 * N_OBS
        for i in range(N_PERSONS):
            base     = person_start + 3 * i
            detected = obs[base + 2]
            if detected < 0.5:
                obs[base]     = 0.0
                obs[base + 1] = 0.0
        return np.concatenate([obs, self._coverage_grid], dtype=np.float32)

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _min_known_clearance(self, pos: np.ndarray, known_obs: list) -> float:
        if not known_obs:
            return 1e6
        x, y = float(pos[0]), float(pos[1])
        best = 1e6
        for ox, oy, rad in known_obs:
            d = float(np.hypot(x - ox, y - oy)) - float(rad)
            if d < best:
                best = d
        return best

    def _min_wall_clearance(self, pos: np.ndarray) -> float:
        x, y = float(pos[0]), float(pos[1])
        return min(
            x - R_BODY,
            (MAP_W - R_BODY) - x,
            y - R_BODY,
            (MAP_H - R_BODY) - y,
        )

    @staticmethod
    def _exp_proximity_penalty(
        clearance: float,
        safe_clearance: float,
        alpha: float,
        max_penalty: float,
    ) -> float:
        """Exponentially increasing penalty as clearance drops below threshold."""
        if clearance >= safe_clearance:
            return 0.0
        deficit = (safe_clearance - clearance) / max(safe_clearance, 1e-6)
        deficit = float(np.clip(deficit, 0.0, 1.5))
        if alpha <= 1e-6:
            scaled = deficit
        else:
            denom  = np.expm1(alpha)
            scaled = np.expm1(alpha * deficit) / max(denom, 1e-9)
        scaled = float(np.clip(scaled, 0.0, 1.0))
        return max_penalty * scaled

    # ── Reward ────────────────────────────────────────────────────────────────

    def _shape_reward(
        self,
        info: dict,
        raw_obs: np.ndarray,
        new_coverage_cells: int,
        min_obs_clearance: float,
        min_wall_clearance: float,
    ) -> float:
        reward = 0.0

        # 1. Person detection delta
        current_found  = info['found']
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            reward += R_PERSON_FOUND * new_detections
        self._prev_found = current_found

        # 2. Collision delta
        if info['collisions'] > self._prev_collisions:
            reward += R_COLLISION
        self._prev_collisions = info['collisions']

        # 3. Formation quality bonus
        if info['form_dev_mean'] < 0.5:
            reward += R_FORMATION_BONUS

        # 4. Coverage bonus — sensor-based new cells
        if new_coverage_cells > 0:
            reward += R_COVERAGE * float(new_coverage_cells)

        # 5. Proximity penalties (obstacles & walls)
        reward += self._exp_proximity_penalty(
            min_obs_clearance,
            SAFETY_SLOW_CLEARANCE,
            EXP_OBS_ALPHA,
            R_NEAR_OBS_EXP_MAX,
        )
        reward += self._exp_proximity_penalty(
            min_wall_clearance,
            WALL_SAFE_CLEARANCE,
            EXP_WALL_ALPHA,
            R_NEAR_WALL_EXP_MAX,
        )

        # 6. Time pressure
        reward += R_TIME_PENALTY

        return float(reward)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple:
        if seed is not None:
            self._env.seed = seed
        else:
            self._env.seed = self._base_seed + self._episode_count
        self._episode_count += 1

        self._env.reset()
        self._apply_curriculum()

        self._coverage_grid = np.zeros(COVERAGE_DIM, dtype=np.float32)
        self._update_sensor_coverage()

        self._step_count      = 0
        self._prev_found      = self._env.total_found
        self._prev_collisions = 0
        self._prev_v_l        = 0.0
        self._prev_w_l        = 0.0

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple:
        a = np.clip(action, -1.0, 1.0)
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()

        # Leader RL commands (raw)
        v_l = float((a[0] + 1.0) / 2.0 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)

        # Safety speed limits near obstacles / walls
        min_obs_clearance  = self._min_known_clearance(leader.pos, known_obs)
        min_wall_clearance = self._min_wall_clearance(leader.pos)
        min_clearance      = min(min_obs_clearance, min_wall_clearance)
        if min_clearance < SAFETY_STOP_CLEARANCE:
            v_l = min(v_l, 0.15 * V_MAX)
        elif min_clearance < SAFETY_SLOW_CLEARANCE:
            v_l = min(v_l, 0.50 * V_MAX)

        # Smooth leader commands (EMA)
        v_l = float(COMMAND_SMOOTH_ALPHA * self._prev_v_l
                    + (1.0 - COMMAND_SMOOTH_ALPHA) * v_l)
        w_l = float(COMMAND_SMOOTH_ALPHA * self._prev_w_l
                    + (1.0 - COMMAND_SMOOTH_ALPHA) * w_l)
        v_l = float(np.clip(v_l, 0.0, V_MAX))
        w_l = float(np.clip(w_l, -OMEGA_MAX, OMEGA_MAX))
        self._prev_v_l = v_l
        self._prev_w_l = w_l

        # APF follower commands
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)

        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1

        # Sensor-based coverage update
        new_coverage_cells = self._update_sensor_coverage()

        obs    = self._build_obs(raw_obs)
        reward = self._shape_reward(
            info, raw_obs, new_coverage_cells,
            min_obs_clearance, min_wall_clearance,
        )

        terminated = int(info['found']) >= int(info['total'])
        truncated  = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

        info['coverage_frac'] = (
            float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if not hasattr(self, '_fig'):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor('#12121f')
            plt.ion()

        leader = self._env.robots[0]
        f1_tgt = self._fctrl1.formation_target(leader.pose)
        f2_tgt = self._fctrl2.formation_target(leader.pose)
        self._env.render(self._ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)

        for idx in np.flatnonzero(self._coverage_grid > 0.5):
            ci = idx // GRID_N
            cj = idx  % GRID_N
            self._ax.add_patch(plt.Rectangle(
                (ci * CELL_SIZE_X, cj * CELL_SIZE_Y),
                CELL_SIZE_X, CELL_SIZE_Y,
                linewidth=0, facecolor='#00e676', alpha=0.08, zorder=1,
            ))

        self._fig.suptitle(
            f'SAR Swarm [RL+APF]  t={self._env.t:.1f}s  '
            f'Active={self._n_active_persons}  '
            f'Coverage={int(np.count_nonzero(self._coverage_grid))}/{COVERAGE_DIM}',
            color='#ccc', fontsize=9,
        )
        plt.pause(0.001)

    def close(self) -> None:
        if hasattr(self, '_fig'):
            plt.close(self._fig)


# ===========================================================================
# SECTION 3  |  ENVIRONMENT FACTORY
# ===========================================================================

def make_env(rank: int, base_seed: int = SEED):
    def _init() -> gym.Env:
        env = SARGymnasiumWrapper(seed=base_seed + rank * 1000)
        return Monitor(env)
    return _init


# ===========================================================================
# SECTION 4  |  CURRICULUM CALLBACK
# ===========================================================================

class CurriculumCallback(BaseCallback):
    """
    Advances curriculum stages based on total training timesteps.
    The eval environment is NOT updated — it always runs at full difficulty.
    """

    def __init__(self, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._current_stage = -1

    def _on_step(self) -> bool:
        target_stage = 0
        for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
            if self.num_timesteps >= threshold:
                target_stage = i

        if target_stage != self._current_stage:
            self._current_stage = target_stage
            n_persons = CURRICULUM_STAGES[target_stage][1]
            self.training_env.env_method('set_n_active_persons', n_persons)

            if self.verbose >= 1:
                stars = '★' * (target_stage + 1)
                print(
                    f"\n{stars}  Curriculum stage {target_stage + 1}"
                    f"/{len(CURRICULUM_STAGES)}  —  "
                    f"{n_persons} active persons  "
                    f"(t={self.num_timesteps:,})  {stars}\n"
                )
        return True


# ===========================================================================
# SECTION 5  |  SAR METRICS CALLBACK
# ===========================================================================

class SARMetricsCallback(BaseCallback):
    """
    Log SAR-specific metrics to TensorBoard at every episode boundary.

    Metrics (prefix: sar/)
    ----------------------
    sar/persons_found    -- fraction of active targets found [0,1]
    sar/collisions       -- collision count for the episode
    sar/form_dev_mean    -- mean follower deviation [m]
    sar/coverage_frac    -- fraction of grid cells visited [0,1]
    sar/curriculum_stage -- current stage (1-4)
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get('infos', []),
            self.locals.get('dones', []),
        ):
            if done and 'found' in info:
                total = max(info.get('total', 1), 1)
                stage = 1
                for i, (thr, _) in enumerate(CURRICULUM_STAGES):
                    if self.num_timesteps >= thr:
                        stage = i + 1
                self.logger.record('sar/persons_found',
                                   info['found'] / total)
                self.logger.record('sar/collisions',
                                   float(info['collisions']))
                self.logger.record('sar/form_dev_mean',
                                   info['form_dev_mean'])
                self.logger.record('sar/coverage_frac',
                                   info.get('coverage_frac', 0.0))
                self.logger.record('sar/curriculum_stage', float(stage))
        return True


# ===========================================================================
# SECTION 6  |  TRAINING FUNCTION
# ===========================================================================

def train(device: str = "auto") -> None:
    """
    Set up environments, build RecurrentPPO, attach callbacks, train.

    TensorBoard key curves
    ----------------------
      sar/persons_found    -> should reach 1.0 by stage 4
      sar/coverage_frac    -> should grow toward 0.8-1.0
      sar/curriculum_stage -> staircase of stage transitions
      sar/collisions       -> should decrease as policy matures
    """
    print("=" * 65)
    print("  SAR SWARM  —  RecurrentPPO + LSTM  (v4)")
    print("=" * 65)
    print(f"  Architecture    : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']})"
          f" + dense {POLICY_KWARGS['net_arch']}")
    print(f"  Action space    : 2-D leader only (APF followers)")
    print(f"  Obs dim         : {SARGymnasiumWrapper.OBS_DIM} "
          f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM})")
    print(f"  Workers         : {N_ENVS}")
    print(f"  Total steps     : {TOTAL_TIMESTEPS:,}")
    print(f"  Device          : {device}")
    print(f"  Curriculum:")
    for i, (thr, n) in enumerate(CURRICULUM_STAGES):
        print(f"    Stage {i+1}: {n:>2} persons  (from step {thr:>9,})")
    print("=" * 65)

    # 1. Env check
    print("\n[1/5] Checking environment wrapper...")
    _check = SARGymnasiumWrapper(seed=SEED)
    check_env(_check, warn=True)
    _check.close()
    print("      ✅  Wrapper OK.\n")

    # 2. Training environments
    print("[2/5] Spawning training environments...")
    train_env = SubprocVecEnv(
        [make_env(rank=i, base_seed=SEED) for i in range(N_ENVS)],
        start_method=_MP_START_METHOD,
    )
    train_env = VecMonitor(train_env)
    train_env.env_method('set_n_active_persons', CURRICULUM_STAGES[0][1])
    print(f"      ✅  {N_ENVS} workers ready.\n")

    # 3. Eval environment (always full difficulty)
    print("[3/5] Building evaluation environment (10 persons, fixed seed)...")
    eval_env = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)),
        n_envs=1,
    )
    eval_env.env_method('set_n_active_persons', N_PERSONS)
    print("      ✅  Eval environment ready (full difficulty).\n")

    # 4. RecurrentPPO model
    print("[4/5] Instantiating RecurrentPPO model...")
    model = RecurrentPPO(
        policy          = "MlpLstmPolicy",
        env             = train_env,
        learning_rate   = LEARNING_RATE,
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = GAMMA,
        gae_lambda      = GAE_LAMBDA,
        clip_range      = CLIP_RANGE,
        ent_coef        = ENT_COEF,
        vf_coef         = VF_COEF,
        max_grad_norm   = MAX_GRAD_NORM,
        policy_kwargs   = POLICY_KWARGS,
        tensorboard_log = TENSORBOARD_LOG,
        verbose         = 1,
        seed            = SEED,
        device          = device,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"      ✅  RecurrentPPO built. Parameters: {n_params:,}\n")

    # 5. Callbacks
    print("[5/5] Configuring callbacks...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        CurriculumCallback(verbose=1),
        SARMetricsCallback(verbose=0),
        EvalCallback(
            eval_env             = eval_env,
            best_model_save_path = CHECKPOINT_DIR,
            log_path             = CHECKPOINT_DIR,
            eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes      = EVAL_EPISODES,
            deterministic        = True,
            render               = False,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path   = CHECKPOINT_DIR,
            name_prefix = "ppo_sar",
            verbose     = 1,
        ),
    ]
    print(f"      ✅  {len(callbacks)} callbacks attached.\n")

    # Train
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  TensorBoard:  tensorboard --logdir ppo_sar_tensorboard\n")
    t0 = time.time()
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = True,
        tb_log_name         = "RecurrentPPO_SAR",
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed:.0f} s)")
    model.save(MODEL_PATH)
    print(f"✅  Final model saved to '{MODEL_PATH}.zip'")
    train_env.close()
    eval_env.close()
    print("\nTo evaluate:  python train_rl.py --mode eval\n")


# ===========================================================================
# SECTION 7  |  EVALUATION FUNCTION
# ===========================================================================

def evaluate(
    n_episodes: int = N_EVAL_EPISODES,
    device: str = "auto",
) -> None:
    """
    Load the saved model and run n_episodes visualised test episodes.
    LSTM state is carried between steps and reset at episode boundaries.
    """
    model_file = f"{MODEL_PATH}.zip"
    if not os.path.exists(model_file):
        print(f"❌  Model '{model_file}' not found.")
        print("   Run  python train_rl.py --mode train  first.")
        return

    print(f"\nLoading model from '{model_file}'...")
    model = RecurrentPPO.load(model_file, device=device)
    print("✅  Model loaded.\n")
    print("=" * 65)
    print(f"  EVALUATION — {n_episodes} episodes  (full difficulty: 10 persons)")
    print("=" * 65)

    results = []

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes}  (seed={SEED + ep}) ---")

        env = SARGymnasiumWrapper(seed=SEED + ep)
        env.set_n_active_persons(N_PERSONS)
        obs, _ = env.reset()

        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)
        cumulative_rew = 0.0
        done           = False
        step_num       = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state         = lstm_states,
                episode_start = episode_starts,
                deterministic = True,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)
            cumulative_rew += reward
            step_num       += 1
            if step_num % 3 == 0:
                env.render()
            done = terminated or truncated

        end_reason = "SUCCESS" if info['found'] >= info['total'] else "TIMEOUT"
        results.append({
            'episode':       ep + 1,
            'reward':        cumulative_rew,
            'persons_found': info['found'],
            'persons_total': info['total'],
            'collisions':    info['collisions'],
            'form_dev_mean': info['form_dev_mean'],
            'coverage_frac': info.get('coverage_frac', 0.0),
            'steps':         step_num,
            'end_reason':    end_reason,
        })

        print(f"  End condition   : {end_reason}")
        print(f"  Steps           : {step_num} / {MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {info['found']} / {info['total']}")
        print(f"  Collisions      : {info['collisions']}")
        print(f"  Form. dev (avg) : {info['form_dev_mean']:.3f} m")
        print(f"  Map coverage    : {info.get('coverage_frac', 0.0)*100:.1f}%")
        input("  [Press Enter for next episode] ")
        env.close()

    # Summary
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  {'Ep':>3}  {'Reward':>9}  {'Found':>8}  "
          f"{'Coll':>5}  {'Coverage':>9}  {'End':>9}")
    print("  " + "-" * 60)
    for r in results:
        found_str = f"{r['persons_found']}/{r['persons_total']}"
        print(f"  {r['episode']:>3}  {r['reward']:>9.1f}  {found_str:>8}  "
              f"{r['collisions']:>5}  "
              f"{r['coverage_frac']*100:>7.1f}%    "
              f"{r['end_reason']:>9}")

    rewards   = [r['reward']          for r in results]
    found_f   = [r['persons_found'] / max(r['persons_total'], 1) for r in results]
    coverages = [r['coverage_frac']   for r in results]
    print("  " + "-" * 60)
    print(f"  Mean reward      : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Mean rescue rate : {np.mean(found_f) * 100:.1f}%")
    print(f"  Mean coverage    : {np.mean(coverages) * 100:.1f}%")
    print("=" * 65)


# ===========================================================================
# SECTION 8  |  ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RecurrentPPO (LSTM) training for the SAR Swarm."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both", "check"],
        default="both",
        help="train | eval | both (default) | check",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EVAL_EPISODES,
        help=f"Eval episodes (default: {N_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for training/evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "check":
        print("Running SB3 environment checker...")
        env = SARGymnasiumWrapper(seed=SEED)
        check_env(env, warn=True)
        env.close()
        print("✅  check_env() passed.")

    elif args.mode == "train":
        train(device=args.device)

    elif args.mode == "eval":
        evaluate(n_episodes=args.episodes, device=args.device)

    elif args.mode == "both":
        train(device=args.device)
        evaluate(n_episodes=args.episodes, device=args.device)
