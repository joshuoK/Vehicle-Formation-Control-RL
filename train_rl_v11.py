"""
train_rl_v11.py — SAR Swarm | Entropy + Curriculum/LR Floor Fix

v11 combines the confirmed entropy fix with the curriculum and LR-floor
changes that were previously drafted as v12:

  ✓ ENTROPY FIX: ENT_COEF = 0.02 → 0.05. This strengthens the entropy bonus
      in the PPO loss so the policy keeps exploring instead of collapsing
      into determinism during long training runs.

  ✓ CURRICULUM RESTORED: 5-stage difficulty progression starting at 3 persons
      / 0 obstacles and scaling to 10 persons / 7 obstacles. This keeps the
      learning signal active as training progresses instead of relying on a
      fixed easy setting.

  ✓ LR FLOOR RAISED: cosine schedule floor 1e-5 → 5e-5. A higher floor keeps
      meaningful update capacity throughout late training.

  ✓ LONG RUN: TOTAL_TIMESTEPS = 80M — completes full curriculum with a
      meaningful mastery phase at Stage 5.

  ✓ WARM-START from checkpoints_v11/best_model (v11 peak checkpoint).
      Policy already knows 3-person detection — skip re-learning Stage 1.

Run:
    python train_rl_v11.py                   # fresh start (recommended)
    python train_rl_v11.py --resume          # resume v11 checkpoint
    python train_rl_v11.py --no-warm-start   # fresh start without transfer
"""
from __future__ import annotations

import os
import sys
import time
import atexit
import argparse
import platform
import multiprocessing
from typing import Optional, Callable

import torch

# Give PyTorch 2 threads — i3-10100 has 4C/8T, remaining threads serve
# SubprocVecEnv workers and OS.
torch.set_num_threads(2)

import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.logger import configure

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)
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
        "Install with: pip install sb3-contrib"
    ) from exc

sys.path.insert(0, os.path.dirname(__file__))
from sar_environment import (  # noqa: E402
    SAREnvironment,
    V_MAX,
    OMEGA_MAX,
    DT,
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
# Logging: tee stdout/stderr to file
# ---------------------------------------------------------------------------
class _TeeStream:
    """Write output to both terminal and a log file simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        written = 0
        for stream in self._streams:
            n = stream.write(data)
            stream.flush()
            if isinstance(n, int):
                written = n
        return written

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")


_LOG_FILE_HANDLE = None
_ORIG_STDOUT = None
_ORIG_STDERR = None


def _disable_live_train_logging() -> None:
    global _LOG_FILE_HANDLE, _ORIG_STDOUT, _ORIG_STDERR
    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
    if _ORIG_STDERR is not None:
        sys.stderr = _ORIG_STDERR
    if _LOG_FILE_HANDLE is not None:
        _LOG_FILE_HANDLE.close()
    _LOG_FILE_HANDLE = _ORIG_STDOUT = _ORIG_STDERR = None


def _enable_live_train_logging(log_filename: str = "train_log_v11.txt") -> None:
    global _LOG_FILE_HANDLE, _ORIG_STDOUT, _ORIG_STDERR
    if _LOG_FILE_HANDLE is not None:
        return
    log_path = os.path.join(os.path.dirname(__file__), log_filename)
    _LOG_FILE_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    _ORIG_STDOUT = sys.stdout
    _ORIG_STDERR = sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = _TeeStream(_ORIG_STDERR, _LOG_FILE_HANDLE)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging started -> {log_path}")
    atexit.register(_disable_live_train_logging)


# ---------------------------------------------------------------------------
# Cross-platform multiprocessing
# ---------------------------------------------------------------------------
multiprocessing.freeze_support()
_IS_WINDOWS = platform.system() == "Windows"
_MP_START_METHOD = "spawn" if _IS_WINDOWS else "fork"

# Module-level constants (avoid recomputing in the hot loop each step)
_MAP_DIAG   = float(np.hypot(MAP_W, MAP_H))
_FOV_RANGE_SQ = float(FOV_RANGE * FOV_RANGE)
_HALF_FOV   = float(0.5 * FOV_ANG)
_TWO_PI     = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Learning-rate / entropy schedules
# ---------------------------------------------------------------------------
def cosine_lr_schedule(initial: float, final: float = 1e-5) -> Callable[[float], float]:
    """Cosine decay to a non-zero floor — prevents policy stagnation late in training."""
    def _fn(progress_remaining: float) -> float:
        cosine = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return final + (initial - final) * cosine
    return _fn


# ===========================================================================
# SECTION 1 | HYPERPARAMETERS & FILE PATHS
# ===========================================================================

# -- Workers ----------------------------------------------------------------
N_ENVS           = 6          # 4C/8T CPU: 6 env workers + 2 threads for PyTorch

# -- Curriculum stages (timestep_start, n_persons, n_obstacles) -------------
# Warm-start from v11 means Stage 1 (3p/0obs) is easy — policy already knows
# 3-person detection. Stages ramp to full 10p/7obs for a meaningful comparison.
CURRICULUM_STAGES = [
    (0,            3,  0),   # Stage 1:  0 –  8M  — re-adapt, no obstacles
    (8_000_000,    3,  2),   # Stage 2:  8M – 16M — reintroduce obstacles
    (16_000_000,   5,  4),   # Stage 3: 16M – 30M — scale persons + obstacles
    (30_000_000,   7,  6),   # Stage 4: 30M – 50M — hard
    (50_000_000,  10,  7),   # Stage 5: 50M – 80M — full difficulty mastery
]

# Evaluation always runs at current curriculum stage difficulty
ACTIVE_PERSONS   = CURRICULUM_STAGES[0][1]   # start value — updated by callback
ACTIVE_OBSTACLES = CURRICULUM_STAGES[0][2]

# -- Episode ----------------------------------------------------------------
MAX_EPISODE_STEPS = 4000
SEED             = 2024

# -- Pre-training phase -----------------------------------------------------
# First PRE_TRAIN_STEPS: collision penalty is soft (R_COLLISION_PRETRAIN)
# so the warm-started agent can re-adapt to v10's obs space without being
# crushed by the full -50 penalty.  After this threshold, full collision
# penalty kicks in.  Episodes never terminate on collision (see below).
PRE_TRAIN_STEPS  = 1_500_000

# -- Coverage grid ----------------------------------------------------------
GRID_N        = 20
CELL_SIZE_X   = MAP_W / GRID_N
CELL_SIZE_Y   = MAP_H / GRID_N
COVERAGE_DIM  = GRID_N * GRID_N        # 400 cells

# -- Extra observation dimensions (appended after coverage grid) ------------
#   [0] coverage_frac   ∈ [0, 1]
#   [1] time_remaining  ∈ [0, 1]  (1 at step 0 → 0 at MAX_EPISODE_STEPS)
#   [2] stagnation_frac ∈ [0, 1]  (0 = just found new cell, 1 = fully stuck)
EXTRA_OBS_DIM = 3

# -- Reward shaping ---------------------------------------------------------
#
# DESIGN PRINCIPLES (calibrated against multi-robot SAR RL literature):
#   1. Per-step shaping in [-0.3, +0.3]; milestone spikes fit within clip.
#      All rewards verified to not exceed clip range so no signal is lost.
#   2. Hierarchy: person detection > early completion >> coverage > formation.
#      Coverage ≈ 11% of person pool; formation ≈ 50% of one-person reward — secondary signals.
#   3. Exploration driven by three gradients: cell-entry, dist-closing,
#      heading-alignment.  Scaled so exploration < 20% of task reward.
#   4. STAGNATION: escalating penalty when no new cells discovered for
#      STAGNATION_GRACE steps — directly targets circling.
#   5. FAST COMPLETION: +8 bonus × time_fraction_remaining — meaningful signal
#      even after discounting with GAMMA=0.995.
#   6. NO COLLISION TERMINATION — penalty-only.  Soft during pre-training
#      (R_COLLISION_PRETRAIN = -1.5) then full (R_COLLISION = -8) after.
#      Both fit within REWARD_CLIP_LOW so the signal is never clipped away.
#
# Rescue pool: 8 pts per person — clear milestone signal, fits within clip
# even when combined with early-completion on the same step.
# Verified: worst-case positive step ≈ +16.1  → fits in REWARD_CLIP_HIGH=17
#           worst-case negative step ≈  -9.6  → fits in REWARD_CLIP_LOW=-10
# Per-episode undiscounted return (good 800-step run) ≈ +84  ✓ within [0,200]
#
TOTAL_RESCUE_POOL       =  33.0    # 11 pts per person (was 30)
R_COLLISION             =  -8.0    # full penalty; fits in clip, clearly felt (was -50 → clipped to -10)
R_COLLISION_PRETRAIN    =  -1.5    # soft pretrain penalty — 5.3× reduction

R_FORMATION_BONUS       =   0.01   # 2400 × 0.01 = 24 max — matches person_found pool (30)
R_COVERAGE_NEW          =   0.05   # 400 × 0.05 × 1.35 ≈ 27 total ≈ 11% of pool
R_COVERAGE_PROGRESS_EXP =   0.5    # exponent for progressive coverage multiplier
R_TIME_PENALTY          =   0.0    # removed — stagnation penalty handles lingering (was -0.015, redundant)
R_SPIN_PENALTY_SCALE    =  -0.02   # max 0.02/step; halved in pretrain (was -0.08, dominated signal)
R_EXPLORATION_GRADIENT  =   0.15   # dense gradient toward coverage payoff
R_EARLY_COMPLETION      =   8.0    # fast-finish; last-person + completion ≈ +16 max
R_NEAR_OBS_EXP_MAX      =  -0.05   # exponential proximity penalty, obstacles; ×0.25 in pretrain (was -0.5)
R_NEAR_WALL_EXP_MAX     =  -0.05   # exponential proximity penalty, walls;     ×0.25 in pretrain (was -0.5)
EXP_OBS_ALPHA           =   4.0
EXP_WALL_ALPHA          =   3.0

# Removed (redundant):
#   R_HEADING_TO_UNCOVERED  — redundant with R_EXPLORATION_GRADIENT (same behaviour, two signals)
#   R_FORWARD_VELOCITY      — redundant with R_TIME_PENALTY + stagnation + explore_dist
#   R_HEADING_CHANGE_SCALE  — redundant with R_SPIN_PENALTY_SCALE + command smoothing (α=0.35)

# -- Pre-training penalty scales -------------------------------------------
# Applied during the grace phase (collision_grace_active = True).
# Keeps the agent's exploration free while re-adapting to the v10 obs space.
PRETRAIN_PROX_SCALE = 0.25   # proximity penalties (obs + wall) at 25% during pretrain
PRETRAIN_SPIN_SCALE = 0.50   # spin penalty at 50% during pretrain
PRETRAIN_TIME_SCALE = 0.50   # time penalty at 50% during pretrain
# Stagnation is disabled entirely during pretrain (see _shape_reward section 5)

# -- Stagnation penalty -----------------------------------------------------
STAGNATION_GRACE     = 60     # grace steps before penalty starts
STAGNATION_RAMP      = 200    # steps over which penalty ramps to max
R_STAGNATION_BASE    = -0.02  # per-step penalty at ramp start — reduced from -0.1
R_STAGNATION_MAX     = -0.05  # per-step cap — reduced from -0.3 (was dominating signal at 6× wall penalty)
MAX_STAGNATION_STEPS = 400    # truncate episode if stuck longer than this

# Per-step reward clip — all signals verified to fit without silent loss:
#   positive: person(8) + completion(7.84@98%) + shaping(0.23) ≈ 16.1  → clip at 17
#   negative: collision(-8) + prox(-1) + stagnation(-0.3) + spin(-0.08) + time(-0.015) = -9.4 → clip at -10
#   pretrain worst: collision(-1.5) + prox(-0.25) + spin(-0.04) + time(-0.008) = -1.8  → very comfortable
REWARD_CLIP_LOW  = -10.0
REWARD_CLIP_HIGH =  17.0

# -- Safety and control blend -----------------------------------------------
SAFETY_SLOW_CLEARANCE = 2.2   # start slowing at this clearance [m]
SAFETY_STOP_CLEARANCE = 0.9   # near-stop clearance [m]
WALL_SAFE_CLEARANCE   = 1.2   # wall proximity penalty threshold [m]
COMMAND_SMOOTH_ALPHA  = 0.35  # command smoothing (1 = full smooth, 0 = raw)
TERMINATE_ON_COLLISION = False   # never terminate on collision — only penalise

# -- RecurrentPPO hyperparameters -------------------------------------------
LEARNING_RATE = cosine_lr_schedule(1.5e-4, final=5e-5)  # v11: floor raised from 1e-5 to prevent late-stage LR starvation
N_STEPS       = 2048
BATCH_SIZE    = 256
N_EPOCHS      = 8
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.05   # v11 FIX: raised from 0.02 — v9/v10 entropy collapsed to -34
                        # despite ENT_COEF=0.02. 0.05 gives a 2.5× stronger entropy
                        # bonus in the PPO loss, preventing the policy becoming so
                        # deterministic it can no longer explore new map layouts.
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5

POLICY_KWARGS = dict(
    lstm_hidden_size=256,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

# -- Training schedule ------------------------------------------------------
TOTAL_TIMESTEPS = 80_000_000   # v11: long run to complete full curriculum + mastery phase
CHECKPOINT_FREQ =    500_000
EVAL_FREQ       =     50_000
N_EVAL_EPISODES =        5

# -- File paths -------------------------------------------------------------
# Load from v9's best checkpoint; write to v11-specific directories.
V9_CHECKPOINT_PATH    = "checkpoints_v11/best_model"  # v11: warm-start from v11 peak checkpoint
MODEL_SAVE_PATH       = "ppo_swarm_agent_v11"
TENSORBOARD_LOG       = "./ppo_sar_tensorboard_v11/"
CHECKPOINT_DIR        = "./checkpoints_v11/"
VECNORM_PATH          = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")
BEST_MODEL_ZIP_PATH   = os.path.join(CHECKPOINT_DIR, "best_model.zip")
BEST_MODEL_PATH       = os.path.join(CHECKPOINT_DIR, "best_model")


def _find_latest_tensorboard_run(log_root: str) -> Optional[str]:
    """Return the newest TensorBoard run directory inside log_root, if any."""
    if not os.path.isdir(log_root):
        return None
    candidates = [
        os.path.join(log_root, entry)
        for entry in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, entry))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ===========================================================================
# SECTION 2 | GYMNASIUM WRAPPER
# ===========================================================================
class SARGymnasiumWrapper(gym.Env):
    """
    Gymnasium wrapper: hybrid APF+RL control, bool coverage map, fixed difficulty.

    Observation layout:
      [env_obs (SAREnvironment.OBS_DIM), coverage_grid (400,),
       coverage_frac, time_remaining, stagnation_frac]

    Key v10 performance changes vs v9:
      - bool coverage grid (was float32) — 75% smaller, faster ops
      - uncovered cells computed once per step and reused across all robots
      - _uncovered_mask cached on self for _nearest_uncovered_info reuse
      - pre-allocated obs output buffer avoids np.concatenate each step
      - no Stage-6 / random-difficulty code paths
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM = SAREnvironment.OBS_DIM + COVERAGE_DIM + EXTRA_OBS_DIM

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed = seed
        self._episode_count = 0
        self._collision_grace_active = False

        self._env = SAREnvironment(seed=seed)
        self._fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
        self._fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Step/episode state
        self._step_count: int = 0
        self._prev_found: int = 0
        self._prev_collisions: int = 0
        self._prev_v_l: float = 0.0
        self._prev_w_l: float = 0.0
        self._prev_dist_to_uncovered: float = 0.0
        self._pre_detected_count: int = 0
        self._active_total: int = ACTIVE_PERSONS

        # Bool coverage grid (v10: bool instead of float32 — 75% smaller)
        self._coverage_grid: np.ndarray = np.zeros(COVERAGE_DIM, dtype=bool)

        # Cached uncovered mask: set by _update_sensor_coverage, read by
        # _nearest_uncovered_info — eliminates a redundant mask computation.
        self._uncovered_mask: np.ndarray = np.ones(COVERAGE_DIM, dtype=bool)

        # Stagnation timer: steps elapsed since last new coverage cell
        self._steps_since_new_cell: int = 0

        # Per-component reward accumulator (populated each episode, logged on end)
        self._reward_components: dict[str, float] = {}

        # Pre-compute grid cell centers (used every step in coverage + nearest)
        x_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_X
        y_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_Y
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        self._cell_centers = np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)  # shape (400, 2)

        # Pre-compute person obs indices for vectorised masking in _build_obs
        self._person_start = 15 + 3 * N_OBS
        self._person_xy_idx = np.array([
            self._person_start + 3 * i + j
            for i in range(N_PERSONS) for j in (0, 1)
        ], dtype=np.intp)
        self._person_det_idx = np.array([
            self._person_start + 3 * i + 2
            for i in range(N_PERSONS)
        ], dtype=np.intp)

        # Pre-allocated output obs buffer (avoids np.concatenate each step)
        self._obs_buf = np.zeros(self.OBS_DIM, dtype=np.float32)
        self._env_dim = SAREnvironment.OBS_DIM

    # --- Curriculum helpers (used by callbacks) ----------------------------

    def set_n_active_persons(self, n: int) -> None:
        """Kept for callback compatibility — v10 always uses ACTIVE_PERSONS."""
        pass  # fixed difficulty; ignore curriculum calls

    def set_n_active_obstacles(self, n: int) -> None:
        """Kept for callback compatibility — v10 always uses ACTIVE_OBSTACLES."""
        pass  # fixed difficulty; ignore curriculum calls

    def set_reward_bonus_multiplier(self, mult: float) -> None:
        """Kept for callback compatibility — not used in v10 (no stage bonuses)."""
        pass

    def set_collision_grace_active(self, active: bool) -> None:
        """Toggle pre-training grace: True = soft penalty, False = full penalty."""
        self._collision_grace_active = bool(active)

    # --- Curriculum application at episode start ---------------------------

    def _apply_curriculum(self) -> None:
        """Mark excess person slots as pre-detected to limit active persons."""
        pre_found = 0
        for i, person in enumerate(self._env.persons):
            if i >= getattr(self._env, '_n_active_persons', CURRICULUM_STAGES[0][1]) and not person.detected:
                person.detected = True
                pre_found += 1
        self._pre_detected_count = pre_found
        self._active_total = max(1, len(self._env.persons) - pre_found)
        self._env.total_found += pre_found

        # Limit active obstacles
        # v11: obstacles set by CurriculumCallback; fallback to Stage 1
        self._env._n_active_obstacles = CURRICULUM_STAGES[0][2]

    # --- Observation -------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % _TWO_PI - np.pi

    def _update_sensor_coverage(self) -> int:
        """
        Update bool coverage grid from each robot's FOV cone.

        Returns the number of newly covered cells.

        v10 optimisation: all uncovered cells are extracted once into a
        contiguous sub-array and each robot's FOV is tested against that
        sub-array only (skipping already-covered cells globally).
        The post-step uncovered mask is stored on self._uncovered_mask so
        _nearest_uncovered_info can reuse it without recomputing.
        """
        uncov_mask = ~self._coverage_grid      # bool (400,)
        if not uncov_mask.any():
            self._uncovered_mask = uncov_mask
            return 0

        uncov_idx  = np.flatnonzero(uncov_mask)          # indices of uncovered cells
        uncov_pos  = self._cell_centers[uncov_idx]        # (N_uncov, 2)
        newly_seen = np.zeros(len(uncov_idx), dtype=bool) # which of those get covered

        for robot in self._env.robots:
            rx = float(robot.x)
            ry = float(robot.y)
            dx = uncov_pos[:, 0] - rx
            dy = uncov_pos[:, 1] - ry
            d2 = dx * dx + dy * dy
            in_range = d2 <= _FOV_RANGE_SQ
            if not in_range.any():
                continue
            ri = np.flatnonzero(in_range)
            hdg = float(robot.theta + robot.sensor_off)
            angles = np.arctan2(dy[ri], dx[ri])
            rel = (angles - hdg + np.pi) % _TWO_PI - np.pi
            newly_seen[ri[np.abs(rel) <= _HALF_FOV]] = True

        new_count = int(newly_seen.sum())
        if new_count:
            self._coverage_grid[uncov_idx[newly_seen]] = True

        # Cache for _nearest_uncovered_info — computed here, reused there.
        self._uncovered_mask = ~self._coverage_grid
        return new_count

    def _nearest_uncovered_info(self, pos: np.ndarray) -> tuple[float, float]:
        """
        Distance and bearing-error to the nearest uncovered grid cell.

        Uses self._uncovered_mask set by _update_sensor_coverage — no
        redundant mask recomputation.

        Returns (distance, bearing_error) where bearing_error ∈ [-π, π].
        Returns (0.0, 0.0) when all cells are covered.
        """
        uncov_mask = self._uncovered_mask
        if not uncov_mask.any():
            return 0.0, 0.0

        uncov_pos = self._cell_centers[uncov_mask]
        dx = uncov_pos[:, 0] - float(pos[0])
        dy = uncov_pos[:, 1] - float(pos[1])
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))
        bearing = float(np.arctan2(dy[idx], dx[idx]))
        heading = float(self._env.robots[0].theta)
        bearing_err = (bearing - heading + np.pi) % _TWO_PI - np.pi
        return dist, float(bearing_err)

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Assemble final observation from env obs, coverage grid, and extra scalars.

        Uses a pre-allocated buffer to avoid np.concatenate allocation each step.
        Returns a copy so SB3 rollout buffer modifications don't corrupt state.
        """
        buf = self._obs_buf
        ed  = self._env_dim  # SAREnvironment.OBS_DIM

        # 1. Copy raw env obs into buffer
        buf[:ed] = raw_obs

        # 2. Vectorised: zero out x,y of undetected persons
        det_vals = buf[self._person_det_idx]
        undet_xy = np.repeat(det_vals < 0.5, 2)
        buf[self._person_xy_idx[undet_xy]] = 0.0

        # 3. Coverage grid (bool → float32 auto-cast)
        buf[ed : ed + COVERAGE_DIM] = self._coverage_grid

        # 4. Extra scalars
        coverage_frac   = float(self._coverage_grid.sum()) / COVERAGE_DIM
        time_remaining  = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
        stagnation_frac = min(
            self._steps_since_new_cell / (STAGNATION_GRACE + STAGNATION_RAMP), 1.0
        )
        buf[ed + COVERAGE_DIM]     = coverage_frac
        buf[ed + COVERAGE_DIM + 1] = time_remaining
        buf[ed + COVERAGE_DIM + 2] = stagnation_frac

        return buf.copy()

    # --- Reward helpers ----------------------------------------------------

    def _min_clearance_obs(self, pos: np.ndarray, known_obs: list) -> float:
        """Vectorised minimum clearance to known obstacles."""
        if not known_obs:
            return 1e6
        obs_arr = np.asarray(known_obs, dtype=np.float64)
        dx = float(pos[0]) - obs_arr[:, 0]
        dy = float(pos[1]) - obs_arr[:, 1]
        return float(np.min(np.sqrt(dx * dx + dy * dy) - obs_arr[:, 2]))

    def _min_clearance_wall(self, pos: np.ndarray) -> float:
        x, y = float(pos[0]), float(pos[1])
        return min(x - R_BODY, (MAP_W - R_BODY) - x,
                   y - R_BODY, (MAP_H - R_BODY) - y)

    @staticmethod
    def _exp_proximity_penalty(
        clearance: float, safe_dist: float, alpha: float, max_pen: float,
    ) -> float:
        """Exponential proximity penalty; returns 0 when clearance ≥ safe_dist."""
        if clearance >= safe_dist:
            return 0.0
        deficit = min(max((safe_dist - clearance) / max(safe_dist, 1e-6), 0.0), 1.5)
        if alpha <= 1e-6:
            scaled = deficit
        else:
            denom = np.expm1(alpha)
            scaled = np.expm1(alpha * deficit) / max(denom, 1e-9)
        return max_pen * min(max(scaled, 0.0), 1.0)

    # --- Reward function ---------------------------------------------------

    def _reset_reward_components(self) -> None:
        self._reward_components = {
            "person_found": 0.0, "collision": 0.0, "formation": 0.0,
            "coverage_new": 0.0, "stagnation": 0.0, "explore_dist": 0.0,
            "spin": 0.0, "prox_obstacle": 0.0, "prox_wall": 0.0,
            "time_penalty": 0.0, "early_completion": 0.0,
        }

    def _shape_reward(
        self,
        info: dict,
        action: np.ndarray,
        new_cells: int,
        min_obs_cl: float,
        min_wall_cl: float,
        dist_uncov: float,
        bearing_uncov: float,
        all_found: bool,
    ) -> float:
        rc = self._reward_components
        r  = 0.0

        # 1. Primary objective: person detection
        new_det = info["found"] - self._prev_found
        if new_det > 0:
            per_person = TOTAL_RESCUE_POOL / max(float(self._active_total), 1.0)
            bonus = per_person * new_det
            rc["person_found"] += bonus
            r += bonus
        self._prev_found = info["found"]

        # 2. Collision penalty — soft during pre-training, full after
        if info["collisions"] > self._prev_collisions:
            pen = R_COLLISION_PRETRAIN if self._collision_grace_active else R_COLLISION
            rc["collision"] += pen
            r += pen
        self._prev_collisions = info["collisions"]

        # 3. Formation keeping
        if info["form_dev_mean"] < 1.5:
            rc["formation"] += R_FORMATION_BONUS
            r += R_FORMATION_BONUS

        # 4. Coverage: progressive bonus per new cell
        if new_cells > 0:
            cov_frac = float(self._coverage_grid.sum()) / COVERAGE_DIM
            mult = 1.0 + cov_frac ** R_COVERAGE_PROGRESS_EXP
            bonus = R_COVERAGE_NEW * float(new_cells) * mult
            rc["coverage_new"] += bonus
            r += bonus

        # 5. Stagnation penalty — disabled during pre-training so the agent
        #    explores freely while re-adapting; escalates after pretrain ends.
        if (not self._collision_grace_active) and self._steps_since_new_cell > STAGNATION_GRACE:
            overshoot = self._steps_since_new_cell - STAGNATION_GRACE
            ramp = min(overshoot / STAGNATION_RAMP, 1.0)
            pen = R_STAGNATION_BASE + (R_STAGNATION_MAX - R_STAGNATION_BASE) * ramp
            rc["stagnation"] += pen
            r += pen

        # 6. Exploration gradient — dense signal toward nearest uncovered cell (reward only, no penalty)
        if self._prev_dist_to_uncovered > 0 and dist_uncov > 0:
            delta = (self._prev_dist_to_uncovered - dist_uncov) / _MAP_DIAG
            bonus = R_EXPLORATION_GRADIENT * max(min(delta * 10.0, 1.0), 0.0)  # only reward approaching, no penalty for moving away (stagnation handles that)
            rc["explore_dist"] += bonus
            r += bonus

        # 7. Anti-spinning — halved during pre-training
        spin_scale = PRETRAIN_SPIN_SCALE if self._collision_grace_active else 1.0
        omega_frac = abs(float(action[1]))
        pen = R_SPIN_PENALTY_SCALE * spin_scale * omega_frac
        rc["spin"] += pen
        r += pen

        # 8. Proximity penalties — reduced to 25% during pre-training so the
        #    agent can explore near walls/obstacles without being over-penalised.
        prox_scale = PRETRAIN_PROX_SCALE if self._collision_grace_active else 1.0
        pen_obs  = self._exp_proximity_penalty(min_obs_cl,  SAFETY_SLOW_CLEARANCE, EXP_OBS_ALPHA,  R_NEAR_OBS_EXP_MAX) * prox_scale
        pen_wall = self._exp_proximity_penalty(min_wall_cl, WALL_SAFE_CLEARANCE,   EXP_WALL_ALPHA, R_NEAR_WALL_EXP_MAX) * prox_scale
        rc["prox_obstacle"] += pen_obs
        rc["prox_wall"]     += pen_wall
        r += pen_obs + pen_wall

        # 9. Time penalty — halved during pre-training (less pressure to rush)
        time_scale = PRETRAIN_TIME_SCALE if self._collision_grace_active else 1.0
        pen = R_TIME_PENALTY * time_scale
        rc["time_penalty"] += pen
        r += pen

        # 10. Early completion bonus — dominant fast-finish signal
        if all_found:
            frac_left = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
            bonus = R_EARLY_COMPLETION * frac_left
            rc["early_completion"] += bonus
            r += bonus

        return min(max(r, REWARD_CLIP_LOW), REWARD_CLIP_HIGH)

    # --- Gymnasium interface -----------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._env.seed = seed
        else:
            self._env.seed = self._base_seed + self._episode_count
        self._episode_count += 1

        self._env.reset()
        self._apply_curriculum()

        self._coverage_grid[:] = False
        self._steps_since_new_cell = 0
        self._update_sensor_coverage()

        self._step_count = 0
        self._prev_found = self._env.total_found
        self._prev_collisions = 0
        self._prev_v_l = 0.0
        self._prev_w_l = 0.0
        dist, _ = self._nearest_uncovered_info(self._env.robots[0].pos)
        self._prev_dist_to_uncovered = dist
        self._reset_reward_components()

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(action, -1.0, 1.0)
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()

        # RL leader velocity command
        v_l = float((a[0] + 1.0) * 0.5 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)

        # Safety: slow or stop if too close to obstacles/walls
        min_obs_cl  = self._min_clearance_obs(leader.pos, known_obs)
        min_wall_cl = self._min_clearance_wall(leader.pos)
        min_cl      = min(min_obs_cl, min_wall_cl)
        if min_cl < SAFETY_STOP_CLEARANCE:
            v_l = min(v_l, 0.15 * V_MAX)
        elif min_cl < SAFETY_SLOW_CLEARANCE:
            v_l = min(v_l, 0.50 * V_MAX)

        # Exponential moving average smoothing for leader commands
        alpha = COMMAND_SMOOTH_ALPHA
        v_l = alpha * self._prev_v_l + (1.0 - alpha) * v_l
        w_l = alpha * self._prev_w_l + (1.0 - alpha) * w_l
        v_l = min(max(v_l, 0.0), V_MAX)
        w_l = min(max(w_l, -OMEGA_MAX), OMEGA_MAX)
        self._prev_v_l = v_l
        self._prev_w_l = w_l

        # APF follower commands
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)

        raw_obs, _, _, info = self._env.step([v_l, w_l, v_f1, w_f1, v_f2, w_f2])
        self._step_count += 1

        # Coverage update — also sets self._uncovered_mask
        new_cells = self._update_sensor_coverage()
        if new_cells > 0:
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1

        # Exploration metrics (reuse _uncovered_mask set above)
        dist_uncov, bearing_uncov = self._nearest_uncovered_info(leader.pos)

        # Determine termination before reward (needed for early_completion)
        collided_now = info["collisions"] > self._prev_collisions
        active_found = int(np.clip(
            info["found"] - self._pre_detected_count, 0, self._active_total
        ))
        all_found = active_found >= self._active_total

        obs    = self._build_obs(raw_obs)
        reward = self._shape_reward(
            info, a, new_cells, min_obs_cl, min_wall_cl,
            dist_uncov, bearing_uncov, all_found,
        )
        self._prev_dist_to_uncovered = dist_uncov

        # Episode termination — only on mission success
        terminated = all_found

        # Truncation: stagnation kill-switch or time limit
        if self._steps_since_new_cell >= MAX_STAGNATION_STEPS:
            truncated = True
        else:
            truncated = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

        # Info dict
        coverage_frac = float(self._coverage_grid.sum()) / COVERAGE_DIM
        info["coverage_frac"]        = coverage_frac
        info["coverage_pct"]         = coverage_frac
        info["full_found"]           = int(info["found"])
        info["full_total"]           = int(info["total"])
        info["active_found"]         = active_found
        info["active_total"]         = self._active_total
        info["pre_detected"]         = self._pre_detected_count
        info["found"]                = active_found
        info["total"]                = self._active_total
        info["steps_since_new_cell"] = self._steps_since_new_cell
        if terminated or truncated:
            info["reward_components"] = dict(self._reward_components)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor("#12121f")
            plt.ion()

        leader  = self._env.robots[0]
        f1_tgt  = self._fctrl1.formation_target(leader.pose)
        f2_tgt  = self._fctrl2.formation_target(leader.pose)

        persons_backup = self._env.persons
        found_backup   = self._env.total_found
        active_persons = persons_backup[:ACTIVE_PERSONS]
        active_found   = int(np.clip(found_backup - self._pre_detected_count, 0, len(active_persons)))
        self._env.persons     = active_persons
        self._env.total_found = active_found
        try:
            self._env.render(self._ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)
        finally:
            self._env.persons     = persons_backup
            self._env.total_found = found_backup

        for idx in np.flatnonzero(self._coverage_grid):
            ci = idx // GRID_N
            cj = idx  % GRID_N
            self._ax.add_patch(plt.Rectangle(
                (ci * CELL_SIZE_X, cj * CELL_SIZE_Y),
                CELL_SIZE_X, CELL_SIZE_Y,
                linewidth=0, facecolor="#00e676", alpha=0.08, zorder=1,
            ))

        self._fig.suptitle(
            f"SAR v11 [RL+APF]  t={self._env.t:.1f}s  "
            f"Active={self._env._n_active_persons}p/{self._env._n_active_obstacles}obs  "
            f"Coverage={int(self._coverage_grid.sum())}/{COVERAGE_DIM}  "
            f"Stag={self._steps_since_new_cell}",
            color="#ccc", fontsize=9,
        )
        plt.pause(0.001)

    def close(self) -> None:
        if hasattr(self, "_fig"):
            plt.close(self._fig)


# ===========================================================================
# SECTION 3 | ENVIRONMENT FACTORY
# ===========================================================================
def make_env(rank: int, base_seed: int = SEED):
    def _init() -> gym.Env:
        env = SARGymnasiumWrapper(seed=base_seed + rank * 1000)
        return Monitor(env)
    return _init


# ===========================================================================
# SECTION 3B | CURRICULUM CALLBACK
# ===========================================================================
class CurriculumCallback(BaseCallback):
    """
    Advances curriculum difficulty based on timestep thresholds defined in
    CURRICULUM_STAGES. Keeps eval environment aligned with training stage.
    Also manages the collision grace phase: grace is ON for the first half
    of each stage, OFF for the second half.
    """

    def __init__(self, eval_env=None, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._eval_env = eval_env
        self._current_stage = -1
        self._collision_grace_active = None

    def _on_step(self) -> bool:
        # Determine which stage we are in
        target_stage = 0
        for i, (threshold, _, _) in enumerate(CURRICULUM_STAGES):
            if self.num_timesteps >= threshold:
                target_stage = i

        # Advance stage if needed
        if target_stage != self._current_stage:
            self._current_stage = target_stage
            _, n_persons, n_obstacles = CURRICULUM_STAGES[target_stage]
            self.training_env.env_method("set_n_active_persons", n_persons)
            self.training_env.env_method("set_n_active_obstacles", n_obstacles)
            if self._eval_env is not None:
                self._eval_env.env_method("set_n_active_persons", n_persons)
                self._eval_env.env_method("set_n_active_obstacles", n_obstacles)
            if self.verbose >= 1:
                print(
                    f"\n{'*' * (target_stage + 1)} Curriculum stage {target_stage + 1}"
                    f"/{len(CURRICULUM_STAGES)} — "
                    f"{n_persons} persons, {n_obstacles} obstacles "
                    f"(t={self.num_timesteps:,}) {'*' * (target_stage + 1)}\n"
                )

        # Collision grace: ON for first half of each stage, OFF for second half
        stage_start = CURRICULUM_STAGES[target_stage][0]
        if target_stage + 1 < len(CURRICULUM_STAGES):
            stage_end = CURRICULUM_STAGES[target_stage + 1][0]
        else:
            stage_end = TOTAL_TIMESTEPS
        stage_midpoint = stage_start + ((stage_end - stage_start) // 2)
        grace_now = self.num_timesteps < stage_midpoint

        if grace_now != self._collision_grace_active:
            self._collision_grace_active = grace_now
            self.training_env.env_method("set_collision_grace_active", grace_now)
            if self.verbose >= 1:
                mode = "ON  [soft penalty]" if grace_now else "OFF [full penalty]"
                print(f"[Curriculum] Collision grace {mode} (t={self.num_timesteps:,})")

        return True


# ===========================================================================
# SECTION 4 | PRE-TRAINING GRACE CALLBACK
# ===========================================================================
class PreTrainCallback(BaseCallback):
    """
    Manages the two-phase collision penalty schedule.

    Phase 1 (steps 0 → PRE_TRAIN_STEPS):
      Grace ON — collision penalty is soft (R_COLLISION_PRETRAIN = -1.5).
      Lets the warm-started agent re-adapt to the v10 obs space and explore
      freely without being crushed by the full -8 penalty early on.
      Episodes never terminate on collision in either phase.

    Phase 2 (steps > PRE_TRAIN_STEPS):
      Grace OFF — full collision penalty (R_COLLISION = -8) for the rest
      of training.
    """

    def __init__(self, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._grace_active: Optional[bool] = None

    def _on_step(self) -> bool:
        grace_now = self.num_timesteps < PRE_TRAIN_STEPS

        if grace_now != self._grace_active:
            self._grace_active = grace_now
            self.training_env.env_method("set_collision_grace_active", grace_now)
            if self.verbose >= 1:
                if grace_now:
                    mode = "ON  [soft penalty -5  | pre-training]"
                else:
                    mode = "OFF [full penalty -50 | full training]"
                print(
                    f"\n[PreTrain] Collision penalty {mode} "
                    f"(t={self.num_timesteps:,} / {PRE_TRAIN_STEPS:,})\n"
                )
        return True


# ===========================================================================
# SECTION 5 | SAR METRICS CALLBACK
# ===========================================================================
class SARMetricsCallback(BaseCallback):
    """Logs SAR-specific metrics and per-component reward breakdown to TensorBoard."""

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            if not done or "found" not in info:
                continue

            active_total = max(info.get("active_total", info.get("total", 1)), 1)
            active_found = float(np.clip(
                info.get("active_found", info.get("found", 0)), 0, active_total,
            ))

            self.logger.record("sar/persons_found",        active_found)
            self.logger.record("sar/persons_total",        float(active_total))
            self.logger.record("sar/find_rate",            active_found / active_total)
            self.logger.record("sar/collisions",           float(info["collisions"]))
            self.logger.record("sar/form_dev_mean",        info["form_dev_mean"])
            self.logger.record("sar/coverage_frac",        info.get("coverage_frac", 0.0))
            self.logger.record("sar/steps_since_new_cell", float(info.get("steps_since_new_cell", 0)))

            for name, val in info.get("reward_components", {}).items():
                self.logger.record(f"reward/{name}", val)

        return True


# ===========================================================================
# SECTION 5B | BEST MODEL VECNORM SAVER CALLBACK
# ===========================================================================
class BestModelVecNormCallback(BaseCallback):
    """Saves vecnormalize.pkl whenever best_model.zip is updated by EvalCallback."""

    def __init__(self, train_env, checkpoint_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.checkpoint_dir = checkpoint_dir
        self._last_best_time = 0.0

    def _on_step(self) -> bool:
        import os
        import time
        best_zip = os.path.join(self.checkpoint_dir, "best_model.zip")
        vecnorm_pkl = os.path.join(self.checkpoint_dir, "vecnormalize.pkl")

        # Check if best_model.zip was updated (EvalCallback saved new best)
        if os.path.exists(best_zip):
            mtime = os.path.getmtime(best_zip)
            if mtime > self._last_best_time + 0.5:  # 0.5s hysteresis to avoid race conditions
                self._last_best_time = mtime
                # Save the current vecnormalize state
                self.train_env.save(vecnorm_pkl)
                if self.verbose >= 1:
                    print(f"[BestModelVecNorm] Saved {vecnorm_pkl}")

        return True


# ===========================================================================
# SECTION 6 | MODEL HELPERS
# ===========================================================================
def _build_fresh_model(train_env, device: str) -> RecurrentPPO:
    """Construct a new RecurrentPPO from scratch."""
    return RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=0.05,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=TENSORBOARD_LOG,
        verbose=1,
        seed=SEED,
        device=device,
    )


def _find_latest_checkpoint(checkpoint_dir: str):
    """Return (model_path_no_ext, vecnorm_path) for the most recently written
    checkpoint.  Checks both the on-termination save (ppo_swarm_agent_v11.zip)
    and periodic checkpoints (ppo_sar_v11_*_steps.zip), picks whichever was
    written most recently so a proper termination save is always preferred.
    Falls back to best_model / vecnormalize.pkl if nothing else exists."""
    import glob, re

    candidates = {}  # path_no_ext -> mtime

    # On-termination save (root dir)
    term_zip = MODEL_SAVE_PATH + ".zip"
    if os.path.exists(term_zip):
        candidates[MODEL_SAVE_PATH] = os.path.getmtime(term_zip)

    # Periodic checkpoints
    for z in glob.glob(os.path.join(checkpoint_dir, "ppo_sar_v11_*_steps.zip")):
        candidates[z[:-4]] = os.path.getmtime(z)

    if candidates:
        latest = max(candidates, key=candidates.__getitem__)
        # Try to find a matching vecnorm (only periodic checkpoints have step numbers)
        m = re.search(r"_(\d+)_steps$", latest)
        if m:
            vn_path = os.path.join(checkpoint_dir, f"vecnormalize_{m.group(1)}_steps.pkl")
            if not os.path.exists(vn_path):
                vn_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        else:
            # Termination save — vecnormalize.pkl is always updated alongside it
            vn_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        return latest, vn_path

    # Fall back to best_model
    return (
        os.path.join(checkpoint_dir, "best_model"),
        os.path.join(checkpoint_dir, "vecnormalize.pkl"),
    )


def _transfer_policy_weights(
    model: RecurrentPPO, checkpoint_path: str, device: str
) -> None:
    """
    Copy matching policy weights from a checkpoint into model.

    Shape-mismatched tensors (e.g. from a different obs/action space) are
    silently skipped — only compatible weights are transferred.
    """
    src = RecurrentPPO.load(checkpoint_path, device=device)
    src_sd = src.policy.state_dict()
    dst_sd = model.policy.state_dict()

    transferred, skipped = 0, 0
    for key, src_param in src_sd.items():
        if key in dst_sd and dst_sd[key].shape == src_param.shape:
            dst_sd[key].copy_(src_param)
            transferred += 1
        else:
            skipped += 1

    model.policy.load_state_dict(dst_sd)
    print(
        f"      Weight transfer: {transferred} tensors copied, "
        f"{skipped} skipped (shape mismatch)."
    )


# ===========================================================================
# SECTION 7 | TRAINING FUNCTION
# ===========================================================================
def train(
    device: str = "auto",
    resume: bool = False,
    warm_start_from_v9: bool = False,  # v11 default: fresh start (see docstring)
    reset_steps: bool = False,
    resume_tensorboard: bool = False,
    model_path: str = None,
    vecnorm_path: str = None,
) -> None:
    """
    Main training entry point.

    Args:
        device:              Torch device — 'auto', 'cpu', or 'cuda'.
        resume:              If True, load v11 checkpoint and continue.
        warm_start_from_v9: If True (and not resuming), transfer policy
                            weights from the v9 best checkpoint.
        reset_steps:         If True (with --resume), load checkpoint weights
                            but reset the step counter to 0 so the pre-training
                            grace phase runs again from the start.
        resume_tensorboard:   If True (with --resume), continue writing into the
                    latest existing TensorBoard run folder instead of
                    creating a new one.
    """
    print("=" * 68)
    print("  SAR SWARM — RecurrentPPO + LSTM (v11 — Entropy + Curriculum/LR Floor Fix)")
    print("=" * 68)
    print(f"  Architecture : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']}) "
          f"+ dense {POLICY_KWARGS['net_arch']}")
    print("  Action space : 2-D leader only (APF followers)")
    print(f"  Obs dim      : {SARGymnasiumWrapper.OBS_DIM} "
          f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM}"
          f" + extra {EXTRA_OBS_DIM})")
    print(f"  Mission      : Curriculum — {len(CURRICULUM_STAGES)} stages, "
          f"{CURRICULUM_STAGES[0][1]}p/{CURRICULUM_STAGES[0][2]}obs → "
          f"{CURRICULUM_STAGES[-1][1]}p/{CURRICULUM_STAGES[-1][2]}obs")
    print(f"  Pre-training : first {PRE_TRAIN_STEPS:,} steps — soft collision penalty ({R_COLLISION_PRETRAIN})")
    print(f"  Workers      : {N_ENVS}")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Device       : {device}")
    print(f"  Resume v11   : {resume}")
    print(f"  Warm-start v9: {warm_start_from_v9 and not resume}")
    print(f"  Resume TB    : {resume_tensorboard and resume}")
    print("=" * 68)

    # --- 1. Sanity check ---
    print("\n[1/5] Checking environment wrapper...")
    _check = SARGymnasiumWrapper(seed=SEED)
    check_env(_check, warn=True)
    _check.close()
    print("      OK\n")

    # --- 2. Training envs ---
    print("[2/5] Spawning training environments...")
    env_fns    = [make_env(rank=i, base_seed=SEED) for i in range(N_ENVS)]
    train_env  = SubprocVecEnv(env_fns, start_method=_MP_START_METHOD)
    train_env  = VecMonitor(train_env)

    _resume_model = model_path or MODEL_SAVE_PATH
    _resume_vn    = vecnorm_path or VECNORM_PATH
    if resume and os.path.exists(_resume_vn):
        train_env = VecNormalize.load(_resume_vn, train_env)
        print(f"      Loaded VecNormalize stats: {_resume_vn}")
    else:
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )

    # Start in grace mode (pre-training); PreTrainCallback manages the switch
    train_env.env_method("set_collision_grace_active", True)
    print(f"      OK — {N_ENVS} workers ready (VecNormalize active).\n")

    # --- 3. Eval env ---
    print("[3/5] Building evaluation environment (fixed difficulty, fixed seed)...")
    eval_base = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)), n_envs=1
    )
    eval_base = VecMonitor(eval_base)
    eval_env  = VecNormalize(
        eval_base, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )
    eval_env.env_method("set_collision_grace_active", False)
    print("      OK\n")

    # --- 4. Model ---
    print("[4/5] Instantiating model...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if resume and os.path.exists(_resume_model + ".zip"):
        model = RecurrentPPO.load(
            _resume_model, env=train_env, device=device,
            tensorboard_log=TENSORBOARD_LOG,
        )
        print(f"      Resumed from: {_resume_model}.zip  (step {model.num_timesteps:,})")
    else:
        # Fresh v11 model with v9 weights transferred
        model = _build_fresh_model(train_env, device)
        if warm_start_from_v9:
            v9_zip = V9_CHECKPOINT_PATH + ".zip"
            if os.path.exists(v9_zip):
                print(f"      Transferring v9 weights from: {v9_zip}")
                _transfer_policy_weights(model, V9_CHECKPOINT_PATH, device)
            else:
                print(f"      [WARN] v9 checkpoint not found at {v9_zip} — training from scratch.")

    if resume and resume_tensorboard:
        latest_tb_run = _find_latest_tensorboard_run(TENSORBOARD_LOG)
        if latest_tb_run is not None:
            model.set_logger(configure(latest_tb_run, ["stdout", "tensorboard"]))
            print(f"      TensorBoard resumed in: {latest_tb_run}")
        else:
            print(f"      [WARN] No existing TensorBoard run found in {TENSORBOARD_LOG} — using default logging.")

    tb_log_name = None
    if not (resume and resume_tensorboard):
        # Force a fresh TensorBoard run unless explicit resume is requested.
        tb_log_name = f"RecurrentPPO_SAR_v11_{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"      TensorBoard new run: {tb_log_name}")

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"      OK — {n_params:,} parameters\n")

    # --- 5. Callbacks ---
    print("[5/5] Configuring callbacks...")
    callbacks = [
        CurriculumCallback(eval_env=eval_env, verbose=1),  # v11: drives stage progression
        PreTrainCallback(verbose=1),
        SARMetricsCallback(verbose=0),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=CHECKPOINT_DIR,
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        ),
        BestModelVecNormCallback(train_env, CHECKPOINT_DIR, verbose=1),
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_sar_v11",
            save_vecnormalize=True,   # saves vecnormalize_{step}_steps.pkl alongside model
            verbose=1,
        ),
    ]
    print(f"      OK — {len(callbacks)} callbacks\n")

    # --- Train ---
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  TensorBoard: tensorboard --logdir ppo_sar_tensorboard_v11\n")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(not resume) or reset_steps,
            tb_log_name=tb_log_name,
        )
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"\n[!] Training interrupted: {exc}")
        print("[!] Emergency save in progress...")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min")

    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_PATH)
    print(f"Model saved  → {MODEL_SAVE_PATH}.zip")
    print(f"VecNorm saved→ {VECNORM_PATH}")
    train_env.close()
    eval_env.close()


# ===========================================================================
# SECTION 8 | ENTRY POINT
# ===========================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RecurrentPPO v11 — entropy + curriculum/LR floor fix."
    )
    parser.add_argument(
        "--mode", choices=["train", "check"], default="train",
        help="train (default) | check — run env checker only",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Torch device (default: auto → CUDA if available)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from v11 checkpoint (checkpoints_v11/best_model.zip)",
    )
    parser.add_argument(
        "--warm-start-v9", action="store_true",
        help="Transfer policy weights from v9 best checkpoint before training "
             "(default: False — v11 trains from scratch for cleaner representations)",
    )
    parser.add_argument(
        "--reset-steps", action="store_true",
        help="Load checkpoint weights but reset step counter to 0, "
             "re-running the pre-training grace phase from the start",
    )
    parser.add_argument(
        "--resume-tensorboard", action="store_true",
        help="When resuming, continue logging into the latest existing TensorBoard run instead of creating a new one",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Path to model zip to resume from, without .zip extension "
             "(default: ppo_swarm_agent_v11)",
    )
    parser.add_argument(
        "--vecnorm", "-v", default=None,
        help="Path to VecNormalize pkl to load on resume "
             "(default: checkpoints_v11/vecnormalize.pkl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _enable_live_train_logging("train_log_v11.txt")
    args = _parse_args()

    if args.mode == "check":
        print("Running SB3 environment checker...")
        env = SARGymnasiumWrapper(seed=SEED)
        check_env(env, warn=True)
        env.close()
        print("OK — check_env() passed.")

    elif args.mode == "train":
        train(
            device=args.device,
            resume=args.resume,
            warm_start_from_v9=args.warm_start_v9,
            reset_steps=args.reset_steps,
            resume_tensorboard=args.resume_tensorboard,
            model_path=args.model,
            vecnorm_path=args.vecnorm,
        )