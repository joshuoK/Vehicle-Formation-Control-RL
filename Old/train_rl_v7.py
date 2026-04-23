"""
train_rl.py (v7 — Stagnation Penalty, Performance Tuned)

Changes from v6:
  ✓ STAGNATION TIMER replaces visit-count revisit penalty.  Instead of
    tracking how many times each cell is re-observed (which fires constantly
    during normal forward movement), we now count steps since the last NEW
    cell was discovered.  Penalty escalates the longer the agent goes without
    finding new ground — directly targets circling behaviour.
  ✓ REMOVED _visit_counts array — saves memory and per-step compute.
  ✓ SIMPLIFIED _update_sensor_coverage — returns only new_cells count.
  ✓ PRE-COMPUTED FOV constants (FOV_RANGE_SQ, HALF_FOV) as module-level.
  ✓ CACHED _uncovered_centers to avoid re-slicing every step.
  ✓ VECTORISED _build_obs person masking (no Python loop).
  ✓ REDUCED numpy scalar conversions in hot path.
  ✓ HARDWARE TUNING for i3-10100 (4C/8T) + 32GB RAM + RTX 3060 Ti:
      - torch.set_num_threads(2) — leaves cores free for SubprocVecEnv workers
      - Default device = "cuda" for GPU-accelerated gradient updates
      - N_ENVS = 6 — saturates CPU with env stepping while GPU does backprop
      - N_STEPS = 1024 — smaller rollouts = more frequent GPU updates, better
        GPU utilisation on a mid-range card
      - BATCH_SIZE = 256 — fits comfortably in 8GB VRAM with LSTM
  ✓ STAGNATION OBS: replaced coverage_frac with stagnation_frac in the
    observation so the LSTM can "feel" when it's stuck.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import platform
import multiprocessing
from typing import Optional, Callable

import torch

# Give PyTorch 2 threads — i3-10100 has 4C/8T, remaining threads serve
# the SubprocVecEnv workers and OS.
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

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
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

# Pre-compute constants (avoid repeated math in hot loop)
_MAP_DIAG = float(np.hypot(MAP_W, MAP_H))
_FOV_RANGE_SQ = float(FOV_RANGE * FOV_RANGE)
_HALF_FOV = float(0.5 * FOV_ANG)
_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule: LR decays to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def cosine_entropy_schedule(
    initial: float, final: float
) -> Callable[[float], float]:
    """Cosine-annealed entropy coefficient."""
    def func(progress_remaining: float) -> float:
        cosine = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return final + (initial - final) * cosine
    return func


# ===========================================================================
# SECTION 1 | HYPERPARAMETERS & FILE PATHS
# ===========================================================================

# -- Environment (tuned for i3-10100 + 3060 Ti) ----------------------------
N_ENVS = 6                   # 6 env workers on 4C/8T CPU, GPU does backprop
MAX_EPISODE_STEPS = 4000
SEED = 2024

# -- Coverage grid ----------------------------------------------------------
GRID_N = 10
CELL_SIZE_X = MAP_W / GRID_N
CELL_SIZE_Y = MAP_H / GRID_N
COVERAGE_DIM = GRID_N * GRID_N

# -- Extra observation dimensions (appended after coverage grid) ------------
#   [0] = coverage_frac    ∈ [0, 1]
#   [1] = time_remaining   ∈ [0, 1]   (1 at step 0, 0 at MAX_EPISODE_STEPS)
#   [2] = stagnation_frac  ∈ [0, 1]   (0 = just found new cell, 1 = stuck)
EXTRA_OBS_DIM = 3

# -- Curriculum -------------------------------------------------------------
CURRICULUM_STAGES = [
    (0,         3),
    (800_000,   5),
    (2_000_000, 7),
    (3_400_000, 10),
]

# -- Reward shaping (v7 — stagnation-based anti-circling) ------------------
#
# DESIGN PRINCIPLES:
#   1. Per-step rewards in roughly [-3, +3].  Person-found spikes to +10.
#   2. Every undesirable behaviour has an explicit, named penalty.
#   3. Exploration has THREE complementary gradients:
#      a) Cell-entry bonus (discrete)
#      b) Distance-closing bonus (continuous positional)
#      c) Heading-toward-uncovered bonus (continuous rotational)
#   4. STAGNATION replaces revisit:  escalating penalty when no new cells
#      are discovered for many steps.  Directly targets circling without
#      penalising normal forward movement through covered areas.
#   5. Every component is logged individually to TensorBoard.
#
R_PERSON_FOUND          =  10.0
R_COLLISION             =  -5.0
R_FORMATION_BONUS       =   0.3
R_COVERAGE_NEW          =   1.5
R_COVERAGE_PROGRESS_EXP =   0.5
R_TIME_PENALTY          =  -0.08
R_SPIN_PENALTY_SCALE    =  -0.5
R_HEADING_CHANGE_SCALE  =  -0.15
R_EXPLORATION_GRADIENT  =   0.3    # Positional: reward closing distance
R_HEADING_TO_UNCOVERED  =   0.2    # Rotational: reward facing uncovered areas
R_FORWARD_VELOCITY      =   0.1    # Mild bonus for v > 0 (breaks sit-still)
R_EARLY_COMPLETION      =   5.0    # Bonus scaled by (time_remaining / max_steps)
R_NEAR_OBS_EXP_MAX      =  -1.5
R_NEAR_WALL_EXP_MAX     =  -1.5
EXP_OBS_ALPHA           =   4.0
EXP_WALL_ALPHA          =   5.0

# -- Stagnation penalty (replaces revisit) ----------------------------------
#   After STAGNATION_GRACE steps with no new cell, penalty kicks in.
#   Penalty = R_STAGNATION_BASE * (steps_over_grace / STAGNATION_RAMP)
#   Capped at R_STAGNATION_MAX per step.
STAGNATION_GRACE        = 60     # ~3 seconds of grace before penalty starts
STAGNATION_RAMP         = 200    # steps over which penalty ramps to max
R_STAGNATION_BASE       = -0.3   # per-step penalty at ramp start
R_STAGNATION_MAX        = -2.0   # per-step cap so it doesn't overwhelm

# Per-step reward clip to prevent gradient explosions
REWARD_CLIP_LOW  = -10.0
REWARD_CLIP_HIGH =  15.0

# -- Safety and control blend -----------------------------------------------
SAFETY_SLOW_CLEARANCE  = 1.5
SAFETY_STOP_CLEARANCE  = 0.9
WALL_SAFE_CLEARANCE    = 1.2
COMMAND_SMOOTH_ALPHA   = 0.6
TERMINATE_ON_COLLISION = True

# -- RecurrentPPO hyperparameters (tuned for 3060 Ti 8GB VRAM) -------------
LEARNING_RATE  = linear_schedule(3e-4)
N_STEPS        = 1024         # Smaller rollouts → more frequent GPU updates
BATCH_SIZE     = 256          # Fits comfortably in 8GB VRAM with LSTM
N_EPOCHS       = 10
GAMMA          = 0.995
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.02
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5

POLICY_KWARGS = dict(
    lstm_hidden_size=256,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

# -- Training schedule ------------------------------------------------------
TOTAL_TIMESTEPS  = 4_200_000
CHECKPOINT_FREQ  = 500_000
EVAL_FREQ        = 50_000
EVAL_EPISODES    = 5
N_EVAL_EPISODES  = 3

# -- File paths -------------------------------------------------------------
MODEL_PATH       = "ppo_swarm_agent"
TENSORBOARD_LOG  = "./ppo_sar_tensorboard/"
CHECKPOINT_DIR   = "./checkpoints/"
VECNORM_PATH     = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")


# ===========================================================================
# SECTION 2 | GYMNASIUM WRAPPER
# ===========================================================================
class SARGymnasiumWrapper(gym.Env):
    """
    Gymnasium wrapper with hybrid APF+RL control, coverage map, and curriculum.

    v7 changes:
      - Removed _visit_counts array (saves memory + per-step compute)
      - Stagnation timer tracks steps since last new coverage cell
      - Stagnation_frac added to observation vector
      - Vectorised person-slot masking in _build_obs
      - Pre-computed FOV constants used in coverage update

    Observation layout:
      [base_env_obs..., coverage_grid_flags...,
       coverage_frac, time_remaining, stagnation_frac]
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM = SAREnvironment.OBS_DIM + COVERAGE_DIM + EXTRA_OBS_DIM

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed = seed
        self._episode_count = 0
        self._n_active_persons = CURRICULUM_STAGES[0][1]

        self._env = SAREnvironment(seed=seed)
        self._fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
        self._fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # State trackers
        self._step_count: int = 0
        self._prev_found: int = 0
        self._prev_collisions: int = 0
        self._prev_v_l: float = 0.0
        self._prev_w_l: float = 0.0
        self._prev_heading: float = 0.0
        self._prev_dist_to_uncovered: float = 0.0
        self._pre_detected_count: int = 0
        self._active_total: int = N_PERSONS

        # Coverage: binary grid only (visit counts removed in v7)
        self._coverage_grid: np.ndarray = np.zeros(COVERAGE_DIM, dtype=np.float32)

        # Stagnation timer (v7): steps since last new coverage cell
        self._steps_since_new_cell: int = 0

        # Reward component accumulator (for per-episode logging)
        self._reward_components: dict[str, float] = {}

        # Pre-compute cell centers (used by coverage update + nearest-uncovered)
        x_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_X
        y_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_Y
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        self._coverage_cell_centers = np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)

        # Pre-compute person observation indices for vectorised masking
        self._person_start_idx = 15 + 3 * N_OBS
        self._person_xy_indices = np.array([
            self._person_start_idx + 3 * i + j
            for i in range(N_PERSONS) for j in (0, 1)
        ], dtype=np.intp)
        self._person_det_indices = np.array([
            self._person_start_idx + 3 * i + 2
            for i in range(N_PERSONS)
        ], dtype=np.intp)

    # --- Curriculum ---------------------------------------------------------

    def set_n_active_persons(self, n: int) -> None:
        self._n_active_persons = int(np.clip(n, 1, N_PERSONS))

    def _apply_curriculum(self) -> None:
        self._pre_detected_count = 0
        pre_found = 0
        for i, person in enumerate(self._env.persons):
            if i >= self._n_active_persons and not person.detected:
                person.detected = True
                pre_found += 1
        self._pre_detected_count = pre_found
        self._active_total = max(
            1, len(self._env.persons) - self._pre_detected_count
        )
        self._env.total_found += pre_found

    # --- Observation --------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (_TWO_PI) - np.pi

    def _update_sensor_coverage(self) -> int:
        """Update coverage grid using robot FOV cones.

        Returns:
            Number of newly covered cells this step.
        """
        centers = self._coverage_cell_centers
        observed_new = np.zeros(COVERAGE_DIM, dtype=bool)
        # Only check cells that aren't already covered
        uncovered = self._coverage_grid < 0.5

        if not np.any(uncovered):
            return 0

        for robot in self._env.robots:
            rx = float(robot.x)
            ry = float(robot.y)
            dx = centers[:, 0] - rx
            dy = centers[:, 1] - ry
            # Only check uncovered cells (skip already-covered ones)
            candidates = uncovered.copy()
            d2 = dx * dx + dy * dy
            candidates &= (d2 <= _FOV_RANGE_SQ)
            if not np.any(candidates):
                continue

            sensor_hdg = float(robot.theta + robot.sensor_off)
            # Only compute arctan2 for candidate cells
            cand_idx = np.flatnonzero(candidates)
            angles = np.arctan2(dy[cand_idx], dx[cand_idx])
            rel = (angles - sensor_hdg + np.pi) % _TWO_PI - np.pi
            in_cone = np.abs(rel) <= _HALF_FOV
            observed_new[cand_idx[in_cone]] = True

        new_count = int(np.count_nonzero(observed_new))
        if new_count > 0:
            self._coverage_grid[observed_new] = 1.0
        return new_count

    def _nearest_uncovered_info(
        self, pos: np.ndarray
    ) -> tuple[float, float]:
        """Distance and bearing to the nearest uncovered grid cell.

        Returns:
            (distance, bearing_error)  where bearing_error ∈ [-π, π]
            is the difference between leader heading and direction to
            the nearest uncovered cell.  Returns (0, 0) if fully covered.
        """
        uncovered_mask = self._coverage_grid < 0.5
        if not np.any(uncovered_mask):
            return 0.0, 0.0

        uncovered_centers = self._coverage_cell_centers[uncovered_mask]
        dx = uncovered_centers[:, 0] - float(pos[0])
        dy = uncovered_centers[:, 1] - float(pos[1])
        # Use d² for argmin (avoids sqrt for all cells)
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))

        # Bearing to nearest uncovered cell
        bearing = float(np.arctan2(dy[idx], dx[idx]))
        heading = float(self._env.robots[0].theta)
        bearing_err = (bearing - heading + np.pi) % _TWO_PI - np.pi

        return dist, float(bearing_err)

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = raw_obs.copy()

        # Vectorised: mask undetected person x,y to zero
        det_vals = obs[self._person_det_indices]
        undetected_mask = np.repeat(det_vals < 0.5, 2)  # expand to x,y pairs
        obs[self._person_xy_indices[undetected_mask]] = 0.0

        # Extra features
        coverage_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
        time_remaining = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
        # Stagnation fraction: 0 = just found new cell, 1 = max stuck
        stagnation_frac = min(
            self._steps_since_new_cell / (STAGNATION_GRACE + STAGNATION_RAMP),
            1.0,
        )

        extra = np.array(
            [coverage_frac, time_remaining, stagnation_frac], dtype=np.float32
        )

        return np.concatenate([obs, self._coverage_grid, extra], dtype=np.float32)

    # --- Reward helpers (vectorised) ----------------------------------------

    def _min_known_clearance_vec(
        self, pos: np.ndarray, known_obs: list
    ) -> float:
        """Vectorised minimum clearance to known obstacles."""
        if not known_obs:
            return 1e6
        obs_arr = np.array(known_obs, dtype=np.float64)  # shape (N, 3)
        dx = float(pos[0]) - obs_arr[:, 0]
        dy = float(pos[1]) - obs_arr[:, 1]
        dists = np.sqrt(dx * dx + dy * dy) - obs_arr[:, 2]
        return float(np.min(dists))

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
        if clearance >= safe_clearance:
            return 0.0
        deficit = (safe_clearance - clearance) / max(safe_clearance, 1e-6)
        deficit = min(max(deficit, 0.0), 1.5)
        if alpha <= 1e-6:
            scaled = deficit
        else:
            denom = np.expm1(alpha)
            scaled = np.expm1(alpha * deficit) / max(denom, 1e-9)
        scaled = min(max(scaled, 0.0), 1.0)
        return max_penalty * scaled

    # --- Core reward function -----------------------------------------------

    def _reset_reward_components(self) -> None:
        self._reward_components = {
            "person_found": 0.0,
            "collision": 0.0,
            "formation": 0.0,
            "coverage_new": 0.0,
            "stagnation": 0.0,
            "explore_dist": 0.0,
            "explore_heading": 0.0,
            "forward_vel": 0.0,
            "spin": 0.0,
            "heading_jitter": 0.0,
            "prox_obstacle": 0.0,
            "prox_wall": 0.0,
            "time_penalty": 0.0,
            "early_completion": 0.0,
        }

    def _shape_reward(
        self,
        info: dict,
        raw_obs: np.ndarray,
        action: np.ndarray,
        new_coverage_cells: int,
        min_obs_clearance: float,
        min_wall_clearance: float,
        dist_to_uncovered: float,
        bearing_to_uncovered: float,
        all_found: bool,
    ) -> float:
        rc = self._reward_components  # shorthand
        reward = 0.0

        # ── 1. PRIMARY OBJECTIVE: find persons ──
        current_found = info["found"]
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            r = R_PERSON_FOUND * new_detections
            rc["person_found"] += r
            reward += r
        self._prev_found = current_found

        # ── 2. COLLISION ──
        if info["collisions"] > self._prev_collisions:
            rc["collision"] += R_COLLISION
            reward += R_COLLISION
        self._prev_collisions = info["collisions"]

        # ── 3. FORMATION KEEPING ──
        if info["form_dev_mean"] < 0.5:
            rc["formation"] += R_FORMATION_BONUS
            reward += R_FORMATION_BONUS

        # ── 4. COVERAGE — progressive bonus ──
        if new_coverage_cells > 0:
            cov_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
            multiplier = 1.0 + cov_frac ** R_COVERAGE_PROGRESS_EXP
            r = R_COVERAGE_NEW * float(new_coverage_cells) * multiplier
            rc["coverage_new"] += r
            reward += r

        # ── 5. STAGNATION PENALTY (v7 — replaces revisit) ──
        #   Fires after STAGNATION_GRACE steps with no new cell.
        #   Ramps linearly up to R_STAGNATION_MAX over STAGNATION_RAMP steps.
        #   This directly punishes circling/sitting without penalising
        #   normal forward movement through already-covered areas.
        if self._steps_since_new_cell > STAGNATION_GRACE:
            overshoot = self._steps_since_new_cell - STAGNATION_GRACE
            ramp_frac = min(overshoot / STAGNATION_RAMP, 1.0)
            r = R_STAGNATION_BASE + (R_STAGNATION_MAX - R_STAGNATION_BASE) * ramp_frac
            rc["stagnation"] += r
            reward += r

        # ── 6. EXPLORATION GRADIENT — positional ──
        if self._prev_dist_to_uncovered > 0 and dist_to_uncovered > 0:
            delta = (self._prev_dist_to_uncovered - dist_to_uncovered) / _MAP_DIAG
            r = R_EXPLORATION_GRADIENT * min(max(delta * 10.0, -1.0), 1.0)
            rc["explore_dist"] += r
            reward += r

        # ── 7. EXPLORATION GRADIENT — rotational (heading toward uncovered) ──
        if dist_to_uncovered > 0:
            alignment = float(np.cos(bearing_to_uncovered))
            r = R_HEADING_TO_UNCOVERED * alignment
            rc["explore_heading"] += r
            reward += r

        # ── 8. FORWARD VELOCITY bonus ──
        v_fraction = (float(action[0]) + 1.0) * 0.5  # ∈ [0, 1]
        r = R_FORWARD_VELOCITY * v_fraction
        rc["forward_vel"] += r
        reward += r

        # ── 9. ANTI-SPINNING ──
        omega_fraction = abs(float(action[1]))
        r = R_SPIN_PENALTY_SCALE * omega_fraction
        rc["spin"] += r
        reward += r

        # ── 10. HEADING JITTER ──
        heading = float(self._env.robots[0].theta)
        delta_heading = abs(
            (heading - self._prev_heading + np.pi) % _TWO_PI - np.pi
        )
        r = R_HEADING_CHANGE_SCALE * (delta_heading / np.pi)
        rc["heading_jitter"] += r
        reward += r
        self._prev_heading = heading

        # ── 11. PROXIMITY penalties ──
        r_obs = self._exp_proximity_penalty(
            min_obs_clearance, SAFETY_SLOW_CLEARANCE,
            EXP_OBS_ALPHA, R_NEAR_OBS_EXP_MAX,
        )
        r_wall = self._exp_proximity_penalty(
            min_wall_clearance, WALL_SAFE_CLEARANCE,
            EXP_WALL_ALPHA, R_NEAR_WALL_EXP_MAX,
        )
        rc["prox_obstacle"] += r_obs
        rc["prox_wall"] += r_wall
        reward += r_obs + r_wall

        # ── 12. TIME PENALTY ──
        rc["time_penalty"] += R_TIME_PENALTY
        reward += R_TIME_PENALTY

        # ── 13. EARLY COMPLETION BONUS ──
        if all_found:
            frac_remaining = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
            r = R_EARLY_COMPLETION * frac_remaining
            rc["early_completion"] += r
            reward += r

        # Clip total
        reward = min(max(reward, REWARD_CLIP_LOW), REWARD_CLIP_HIGH)
        return reward

    # --- Gymnasium interface ------------------------------------------------

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

        self._coverage_grid = np.zeros(COVERAGE_DIM, dtype=np.float32)
        self._steps_since_new_cell = 0
        self._update_sensor_coverage()

        self._step_count = 0
        self._prev_found = self._env.total_found
        self._prev_collisions = 0
        self._prev_v_l = 0.0
        self._prev_w_l = 0.0
        self._prev_heading = float(self._env.robots[0].theta)
        dist, _ = self._nearest_uncovered_info(self._env.robots[0].pos)
        self._prev_dist_to_uncovered = dist
        self._reset_reward_components()

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(action, -1.0, 1.0)
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()

        # Pure RL leader control
        v_l = float((a[0] + 1.0) * 0.5 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)

        # Safety speed limits
        min_obs_clearance = self._min_known_clearance_vec(leader.pos, known_obs)
        min_wall_clearance = self._min_wall_clearance(leader.pos)
        min_clearance = min(min_obs_clearance, min_wall_clearance)
        if min_clearance < SAFETY_STOP_CLEARANCE:
            v_l = min(v_l, 0.15 * V_MAX)
        elif min_clearance < SAFETY_SLOW_CLEARANCE:
            v_l = min(v_l, 0.50 * V_MAX)

        # Smooth leader commands
        alpha = COMMAND_SMOOTH_ALPHA
        one_minus_alpha = 1.0 - alpha
        v_l = alpha * self._prev_v_l + one_minus_alpha * v_l
        w_l = alpha * self._prev_w_l + one_minus_alpha * w_l
        v_l = min(max(v_l, 0.0), V_MAX)
        w_l = min(max(w_l, -OMEGA_MAX), OMEGA_MAX)
        self._prev_v_l = v_l
        self._prev_w_l = w_l

        # Follower APF controllers
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)

        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1

        # Coverage update (simplified: returns only new cell count)
        new_cells = self._update_sensor_coverage()

        # Update stagnation timer
        if new_cells > 0:
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1

        # Exploration info
        dist_to_uncov, bearing_to_uncov = self._nearest_uncovered_info(leader.pos)

        # Termination check (needed before reward for early_completion)
        collided_now = info["collisions"] > self._prev_collisions
        active_found = int(np.clip(
            info["found"] - self._pre_detected_count, 0, self._active_total,
        ))
        all_found = active_found >= self._active_total

        obs = self._build_obs(raw_obs)
        reward = self._shape_reward(
            info, raw_obs, a,
            new_cells,
            min_obs_clearance, min_wall_clearance,
            dist_to_uncov, bearing_to_uncov,
            all_found,
        )

        # Update exploration tracker AFTER reward
        self._prev_dist_to_uncovered = dist_to_uncov

        # Termination
        terminated = all_found
        if TERMINATE_ON_COLLISION and collided_now:
            terminated = True
        truncated = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

        # Info dict
        coverage_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
        info["coverage_frac"] = coverage_frac
        info["coverage_pct"] = coverage_frac
        info["full_found"] = int(info["found"])
        info["full_total"] = int(info["total"])
        info["active_found"] = active_found
        info["active_total"] = self._active_total
        info["pre_detected"] = self._pre_detected_count
        info["found"] = active_found
        info["total"] = self._active_total
        info["steps_since_new_cell"] = self._steps_since_new_cell

        # Attach per-component reward breakdown on episode end
        if terminated or truncated:
            info["reward_components"] = dict(self._reward_components)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor("#12121f")
            plt.ion()

        leader = self._env.robots[0]
        f1_tgt = self._fctrl1.formation_target(leader.pose)
        f2_tgt = self._fctrl2.formation_target(leader.pose)
        self._env.render(self._ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)

        for idx in np.flatnonzero(self._coverage_grid > 0.5):
            ci = idx // GRID_N
            cj = idx % GRID_N
            self._ax.add_patch(
                plt.Rectangle(
                    (ci * CELL_SIZE_X, cj * CELL_SIZE_Y),
                    CELL_SIZE_X, CELL_SIZE_Y,
                    linewidth=0, facecolor="#00e676", alpha=0.08, zorder=1,
                )
            )

        stag = self._steps_since_new_cell
        self._fig.suptitle(
            "SAR Swarm [RL+APF] "
            f"t={self._env.t:.1f}s "
            f"Active={self._n_active_persons} "
            f"Coverage={int(np.count_nonzero(self._coverage_grid))}/{COVERAGE_DIM} "
            f"Stag={stag}",
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
# SECTION 4 | CURRICULUM CALLBACK
# ===========================================================================
class CurriculumCallback(BaseCallback):
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
            self.training_env.env_method("set_n_active_persons", n_persons)
            if self.verbose >= 1:
                stars = "*" * (target_stage + 1)
                print(
                    f"\n{stars} Curriculum stage {target_stage + 1}"
                    f"/{len(CURRICULUM_STAGES)} — "
                    f"{n_persons} active persons "
                    f"(t={self.num_timesteps:,}) {stars}\n"
                )
        return True


# ===========================================================================
# SECTION 5 | SAR METRICS CALLBACK (with per-component reward logging)
# ===========================================================================
class SARMetricsCallback(BaseCallback):
    """Logs SAR-specific metrics and individual reward components to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            if not done or "found" not in info:
                continue

            active_total = max(
                info.get("active_total", info.get("total", 1)), 1
            )
            active_found = float(np.clip(
                info.get("active_found", info.get("found", 0)), 0, active_total,
            ))
            stage = 1
            for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
                if self.num_timesteps >= threshold:
                    stage = i + 1

            # Core metrics
            self.logger.record("sar/persons_found", active_found)
            self.logger.record("sar/persons_total", float(active_total))
            self.logger.record("sar/find_rate", active_found / active_total)
            self.logger.record("sar/collisions", float(info["collisions"]))
            self.logger.record("sar/form_dev_mean", info["form_dev_mean"])
            self.logger.record("sar/coverage_frac", info.get("coverage_frac", 0.0))
            self.logger.record("sar/curriculum_stage", float(stage))
            self.logger.record("sar/steps_since_new_cell",
                               float(info.get("steps_since_new_cell", 0)))

            # Per-component reward breakdown
            rc = info.get("reward_components", {})
            for name, value in rc.items():
                self.logger.record(f"reward/{name}", value)

        return True


# ===========================================================================
# SECTION 6 | TRAINING FUNCTION
# ===========================================================================
def train(device: str = "auto", resume: bool = False) -> None:
    print("=" * 65)
    print("  SAR SWARM — RecurrentPPO + LSTM (v7 — Stagnation Penalty)")
    print("=" * 65)
    print(
        f"  Architecture : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']}) "
        f"+ dense {POLICY_KWARGS['net_arch']}"
    )
    print("  Action space : 2-D leader only (APF followers)")
    print(
        f"  Obs dim      : {SARGymnasiumWrapper.OBS_DIM} "
        f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM}"
        f" + extra {EXTRA_OBS_DIM})"
    )
    print(f"  Workers      : {N_ENVS}")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Device       : {device}")
    print(f"  Resume       : {resume}")
    print("  Curriculum:")
    for i, (thr, n) in enumerate(CURRICULUM_STAGES):
        print(f"    Stage {i + 1}: {n:>2} persons (from step {thr:>9,})")
    print("=" * 65)

    # --- 1. Check ---
    print("\n[1/5] Checking environment wrapper...")
    _check = SARGymnasiumWrapper(seed=SEED)
    check_env(_check, warn=True)
    _check.close()
    print("      OK — Wrapper check passed.\n")

    # --- 2. Training envs ---
    print("[2/5] Spawning training environments...")
    env_fns = [make_env(rank=i, base_seed=SEED) for i in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns, start_method=_MP_START_METHOD)
    train_env = VecMonitor(train_env)

    # Observation normalisation (standard PPO practice)
    if resume and os.path.exists(VECNORM_PATH):
        train_env = VecNormalize.load(VECNORM_PATH, train_env)
        print("      Loaded VecNormalize stats from checkpoint.")
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,  # We manage reward scale manually
            clip_obs=10.0,
        )

    train_env.env_method("set_n_active_persons", CURRICULUM_STAGES[0][1])
    print(f"      OK — {N_ENVS} workers ready (VecNormalize active).\n")

    # --- 3. Eval env ---
    print("[3/5] Building evaluation environment (10 persons, fixed seed)...")
    eval_base = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)),
        n_envs=1,
    )
    eval_env = VecNormalize(
        eval_base,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,  # Don't update stats during eval
    )
    eval_env.env_method("set_n_active_persons", N_PERSONS)
    print("      OK — Eval environment ready (full difficulty).\n")

    # --- 4. Model ---
    print("[4/5] Instantiating RecurrentPPO model...")
    if resume and os.path.exists(f"{MODEL_PATH}.zip"):
        model = RecurrentPPO.load(
            MODEL_PATH, env=train_env, device=device,
            tensorboard_log=TENSORBOARD_LOG,
        )
        print(f"      Resumed from '{MODEL_PATH}.zip'")
    else:
        model = RecurrentPPO(
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
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=TENSORBOARD_LOG,
            verbose=1,
            seed=SEED,
            device=device,
        )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"      OK — RecurrentPPO built. Parameters: {n_params:,}\n")

    # --- 5. Callbacks ---
    print("[5/5] Configuring callbacks...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        CurriculumCallback(verbose=1),
        SARMetricsCallback(verbose=0),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=CHECKPOINT_DIR,
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_sar",
            verbose=1,
        ),
    ]
    print(f"      OK — {len(callbacks)} callbacks attached.\n")

    # --- Train ---
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  TensorBoard: tensorboard --logdir ppo_sar_tensorboard\n")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not resume,
            tb_log_name="RecurrentPPO_SAR",
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n[!] Training interrupted: {e}")
        print("[!] Emergency save...")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)
    print(f"OK — Model saved to '{MODEL_PATH}.zip'")
    print(f"OK — VecNormalize saved to '{VECNORM_PATH}'")
    train_env.close()
    eval_env.close()
    print("\nTo evaluate: python train_rl.py --mode eval\n")


# ===========================================================================
# SECTION 7 | EVALUATION FUNCTION
# ===========================================================================
def evaluate(
    n_episodes: int = N_EVAL_EPISODES,
    device: str = "auto",
    wait_for_input: bool = True,
) -> None:
    model_file = f"{MODEL_PATH}.zip"
    if not os.path.exists(model_file):
        print(f"Model '{model_file}' not found.")
        print("Run: python train_rl.py --mode train")
        return

    print(f"\nLoading model from '{model_file}'...")
    model = RecurrentPPO.load(model_file, device=device)
    expected_obs_dim = int(model.observation_space.shape[0])
    warned_obs_mismatch = False

    def _adapt_obs_dim(obs_vec: np.ndarray) -> np.ndarray:
        nonlocal warned_obs_mismatch
        current_dim = int(obs_vec.shape[0])
        if current_dim == expected_obs_dim:
            return obs_vec

        if not warned_obs_mismatch:
            print(
                f"[!] Observation size mismatch: env={current_dim}, model={expected_obs_dim}. "
                "Adapting eval observations to model size."
            )
            warned_obs_mismatch = True

        if current_dim > expected_obs_dim:
            return obs_vec[:expected_obs_dim]

        pad = np.zeros((expected_obs_dim - current_dim,), dtype=obs_vec.dtype)
        return np.concatenate([obs_vec, pad], dtype=obs_vec.dtype)

    print("OK — Model loaded.\n")
    print("=" * 65)
    print(f"  EVALUATION — {n_episodes} episodes (full difficulty: 10 persons)")
    print("=" * 65)

    results = []

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} (seed={SEED + ep}) ---")

        env = SARGymnasiumWrapper(seed=SEED + ep)
        env.set_n_active_persons(N_PERSONS)
        obs, _ = env.reset()
        obs = _adapt_obs_dim(obs)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        cumulative_rew = 0.0
        done = False
        step_num = 0

        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            obs = _adapt_obs_dim(obs)
            episode_starts = np.zeros((1,), dtype=bool)
            cumulative_rew += reward
            step_num += 1
            if wait_for_input and step_num % 3 == 0:
                env.render()
            done = terminated or truncated

        # End reason
        if info["found"] >= info["total"]:
            end_reason = "SUCCESS"
        elif info["collisions"] > 0:
            end_reason = "COLLISION"
        else:
            end_reason = "TIMEOUT"

        results.append({
            "episode": ep + 1,
            "reward": cumulative_rew,
            "persons_found": info["found"],
            "persons_total": info["total"],
            "collisions": info["collisions"],
            "form_dev_mean": info["form_dev_mean"],
            "coverage_frac": info.get("coverage_frac", 0.0),
            "steps": step_num,
            "end_reason": end_reason,
            "reward_components": info.get("reward_components", {}),
        })

        print(f"  End condition   : {end_reason}")
        print(f"  Steps           : {step_num} / {MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {info['found']} / {info['total']}")
        print(f"  Collisions      : {info['collisions']}")
        print(f"  Form. dev (avg) : {info['form_dev_mean']:.3f} m")
        print(f"  Map coverage    : {info.get('coverage_frac', 0.0) * 100:.1f}%")

        # Print reward component breakdown
        rc = info.get("reward_components", {})
        if rc:
            print("  Reward breakdown:")
            for name, val in sorted(rc.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(val) > 0.01:
                    print(f"    {name:>20s} : {val:>+9.1f}")

        if wait_for_input:
            input("  [Press Enter for next episode] ")
        env.close()

    # Summary table
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(
        f"  {'Ep':>3}  {'Reward':>9}  {'Found':>8}  "
        f"{'Coll':>5}  {'Coverage':>9}  {'End':>9}"
    )
    print("  " + "-" * 60)
    for r in results:
        found_str = f"{r['persons_found']}/{r['persons_total']}"
        print(
            f"  {r['episode']:>3}  {r['reward']:>9.1f}  {found_str:>8}  "
            f"{r['collisions']:>5}  {r['coverage_frac'] * 100:>7.1f}%    "
            f"{r['end_reason']:>9}"
        )

    rewards = [r["reward"] for r in results]
    found_counts = [r["persons_found"] for r in results]
    total_counts = [r["persons_total"] for r in results]
    coverages = [r["coverage_frac"] for r in results]
    print("  " + "-" * 60)
    print(f"  Mean reward      : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(
        f"  Mean found       : {np.mean(found_counts):.1f}"
        f" / {np.mean(total_counts):.1f}"
    )
    print(f"  Mean coverage    : {np.mean(coverages) * 100:.1f}%")
    print("=" * 65)


# ===========================================================================
# SECTION 8 | ENTRY POINT
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RecurrentPPO (LSTM) training for the SAR Swarm."
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "both", "check"],
        default="both", help="train | eval | both (default) | check",
    )
    parser.add_argument(
        "--episodes", type=int, default=N_EVAL_EPISODES,
        help=f"Eval episodes (default: {N_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"],
        default="auto", help="Torch device (default: auto → uses CUDA if available)",
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Run eval without Enter between episodes",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from saved checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "check":
        print("Running SB3 environment checker...")
        env = SARGymnasiumWrapper(seed=SEED)
        check_env(env, warn=True)
        env.close()
        print("OK — check_env() passed.")

    elif args.mode == "train":
        train(device=args.device, resume=args.resume)

    elif args.mode == "eval":
        evaluate(
            n_episodes=args.episodes,
            device=args.device,
            wait_for_input=not args.no_prompt,
        )

    elif args.mode == "both":
        train(device=args.device, resume=args.resume)
        evaluate(
            n_episodes=args.episodes,
            device=args.device,
            wait_for_input=not args.no_prompt,
        )
