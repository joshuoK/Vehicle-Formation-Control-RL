"""
train_rl_6dof.py (v10 — Full 6DOF Swarm Control)

Changes from v9:
  ✓ ACTION SPACE: Expanded from 2D (leader only) to 6D (v_l, w_l, v_f1, w_f1, v_f2, w_f2).
  ✓ APF REMOVED: Followers are now entirely driven by the RL policy.
  ✓ SURGICAL WEIGHT TRANSFER: Loads features extractor, MLP, and LSTM weights from
        v9 checkpoint, but skips action_net and log_std to accommodate the shape mismatch (2 -> 6).
  ✓ TIMESTEPS: Set to 13,000,000 (~18 hours of training).
"""
from __future__ import annotations

import os
import sys
import time
import json
import atexit
import argparse
import platform
import multiprocessing
from typing import Optional, Callable

import torch

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
from sar_environment import (
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
    body_to_world,  # Needed for render targets now that APF is gone
)


class _TeeStream:
    def __init__(self, console_stream, log_stream):
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data: str) -> int:
        # Always keep full output in terminal.
        written = self._console_stream.write(data)
        self._console_stream.flush()

        # Skip transient tqdm/progress refreshes in file logs.
        lower = data.lower()
        is_progress_refresh = ("\r" in data) or ("remaining" in lower and "it/s" in lower)
        if not is_progress_refresh:
            self._log_stream.write(data)
            self._log_stream.flush()

        return written if isinstance(written, int) else 0

    def flush(self) -> None:
        self._console_stream.flush()
        self._log_stream.flush()

    def isatty(self) -> bool:
        return getattr(self._console_stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._console_stream, "encoding", "utf-8")


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
    _LOG_FILE_HANDLE = None
    _ORIG_STDOUT = None
    _ORIG_STDERR = None


def _enable_live_train_logging(log_filename: str = "train_log_6dof.txt") -> None:
    global _LOG_FILE_HANDLE, _ORIG_STDOUT, _ORIG_STDERR
    if _LOG_FILE_HANDLE is not None:
        return

    log_path = os.path.join(os.path.dirname(__file__), log_filename)
    # Overwrite on each run so the log mirrors the latest session only.
    _LOG_FILE_HANDLE = open(log_path, "w", encoding="utf-8", buffering=1)
    _ORIG_STDOUT = sys.stdout
    _ORIG_STDERR = sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = _TeeStream(_ORIG_STDERR, _LOG_FILE_HANDLE)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging started -> {log_path}")
    atexit.register(_disable_live_train_logging)

multiprocessing.freeze_support()
_IS_WINDOWS = platform.system() == "Windows"
_MP_START_METHOD = "spawn" if _IS_WINDOWS else "fork"

_MAP_DIAG = float(np.hypot(MAP_W, MAP_H))
_FOV_RANGE_SQ = float(FOV_RANGE * FOV_RANGE)
_HALF_FOV = float(0.5 * FOV_ANG)
_TWO_PI = 2.0 * np.pi


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def cosine_lr_schedule(initial: float, final: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        cosine = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return final + (initial - final) * cosine
    return func


# ===========================================================================
# SECTION 1 | HYPERPARAMETERS & FILE PATHS
# ===========================================================================
N_ENVS = 6
MAX_EPISODE_STEPS = 4000
SEED = 2024

GRID_N = 20
CELL_SIZE_X = MAP_W / GRID_N
CELL_SIZE_Y = MAP_H / GRID_N
COVERAGE_DIM = GRID_N * GRID_N
EXTRA_OBS_DIM = 3

# Unified Curriculum: (Step_Threshold, Persons, Obstacles)
CURRICULUM_STAGES = [
    (0,           3, 0),
    (4_000_000,   3, 2),
    (8_000_000,   5, 4),
    (10_500_000,  7, 6),
    (12_000_000, 10, 7),
]

EVAL_ACTIVE_PERSONS = CURRICULUM_STAGES[-1][1]
EVAL_ACTIVE_OBSTACLES = CURRICULUM_STAGES[-1][2]

# -- Reward shaping (v11 -- Mastery Tuning) -----------
TOTAL_RESCUE_POOL       = 388.0
R_COLLISION             = -350.0
R_FORMATION_BONUS       =   0.08
R_FORMATION_PENALTY     =  -0.12
R_FOLLOWER_APPROACH     =   0.15   # reward followers for closing distance to formation slot
R_COVERAGE_NEW          =   0.8
R_COVERAGE_PROGRESS_EXP =   0.5
R_TIME_PENALTY          =  -0.04
R_SPIN_PENALTY_SCALE    =  -0.15
R_HEADING_CHANGE_SCALE  =  -0.10
R_EXPLORATION_GRADIENT  =   0.3
R_HEADING_TO_UNCOVERED  =   0.10
R_FORWARD_VELOCITY      =   0.1
R_EARLY_COMPLETION      =  400.0
R_NEAR_OBS_EXP_MAX      =  -0.5
R_NEAR_WALL_EXP_MAX     =  -0.7
EXP_OBS_ALPHA           =   4.0
EXP_WALL_ALPHA          =   3.0

STAGNATION_GRACE        = 60
STAGNATION_RAMP         = 200
R_STAGNATION_BASE       = -0.1
R_STAGNATION_MAX        = -0.5
MAX_STAGNATION_STEPS    = 400

REWARD_CLIP_LOW  = -350.0
REWARD_CLIP_HIGH =  500.0

SAFETY_SLOW_CLEARANCE  = 2.2
SAFETY_STOP_CLEARANCE  = 1.3
WALL_SAFE_CLEARANCE    = 1.2
COMMAND_SMOOTH_ALPHA   = 0.35
TERMINATE_ON_COLLISION = True

LEARNING_RATE  = cosine_lr_schedule(1.5e-4, final=1e-5)
N_STEPS        = 2048
BATCH_SIZE     = 256
N_EPOCHS       = 6
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

# 6DOF Paths and Length
TOTAL_TIMESTEPS  = 100_000_000
CHECKPOINT_FREQ  = 500_000
EVAL_FREQ        = 50_000
EVAL_EPISODES    = 5
N_EVAL_EPISODES  = 100

V9_BEST_MODEL_PATH = "checkpoints/best_model" # Best v9 checkpoint
MODEL_SAVE_PATH  = "ppo_swarm_agent_6dof"
TENSORBOARD_LOG  = "./ppo_sar_tensorboard_6dof/"
CHECKPOINT_DIR   = "./checkpoints_6dof/"
LOG_DIR          = "./logs_6dof/"
VECNORM_PATH     = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")
CURRICULUM_STATE_PATH = os.path.join(CHECKPOINT_DIR, "curriculum_state.json")
RANDOM_STAGE4_PERSONS = np.array([3, 5, 7, 10], dtype=np.int32)
RANDOM_STAGE4_WEIGHTS = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
RANDOM_STAGE4_RECOVERY_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15], dtype=np.float64)
NO_REGRESSION_WINDOW_EPISODES = 40
NO_REGRESSION_ENTER_FLOOR = 0.75
NO_REGRESSION_EXIT_FLOOR = 0.90
BEST_SWEEP_EPISODES = 3


# ===========================================================================
# SECTION 2 | GYMNASIUM WRAPPER (6DOF)
# ===========================================================================
class SARGymnasiumWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}
    OBS_DIM = SAREnvironment.OBS_DIM + COVERAGE_DIM + EXTRA_OBS_DIM

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed = seed
        self._episode_count = 0
        self._n_active_persons = CURRICULUM_STAGES[0][1]
        self._random_difficulty = False
        self._stage4_weights = RANDOM_STAGE4_WEIGHTS.astype(np.float64).copy()

        self._env = SAREnvironment(seed=seed)
        
        # 6DOF Observation Space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # 6DOF Action Space: [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._step_count: int = 0
        self._prev_found: int = 0
        self._prev_collisions: int = 0
        self._prev_v_l: float = 0.0
        self._prev_w_l: float = 0.0
        self._prev_v_f1: float = 0.0
        self._prev_w_f1: float = 0.0
        self._prev_v_f2: float = 0.0
        self._prev_w_f2: float = 0.0
        self._prev_heading: float = 0.0
        self._prev_dist_to_uncovered: float = 0.0
        self._prev_f1_form_dist: float = 0.0
        self._prev_f2_form_dist: float = 0.0
        self._pre_detected_count: int = 0
        self._active_total: int = N_PERSONS

        self._coverage_grid: np.ndarray = np.zeros(COVERAGE_DIM, dtype=np.float32)
        self._steps_since_new_cell: int = 0
        self._reward_components: dict[str, float] = {}

        x_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_X
        y_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_Y
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        self._coverage_cell_centers = np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)

        self._person_start_idx = 15 + 3 * N_OBS
        self._person_xy_indices = np.array([
            self._person_start_idx + 3 * i + j
            for i in range(N_PERSONS) for j in (0, 1)
        ], dtype=np.intp)
        self._person_det_indices = np.array([
            self._person_start_idx + 3 * i + 2
            for i in range(N_PERSONS)
        ], dtype=np.intp)

    def set_n_active_persons(self, n: int) -> None:
        self._n_active_persons = int(np.clip(n, 1, N_PERSONS))

    def set_n_active_obstacles(self, n: int) -> None:
        self._env._n_active_obstacles = int(np.clip(n, 0, N_OBS))

    def enable_random_difficulty(self, enabled: bool) -> None:
        self._random_difficulty = bool(enabled)

    def set_stage4_sampling_weights(self, weights: list[float] | np.ndarray) -> None:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if arr.size != RANDOM_STAGE4_PERSONS.size:
            return
        arr = np.clip(arr, 0.0, None)
        total = float(np.sum(arr))
        if total <= 1e-12:
            return
        self._stage4_weights = arr / total

    def _apply_curriculum(self) -> None:
        self._pre_detected_count = 0
        pre_found = 0
        for i, person in enumerate(self._env.persons):
            if i >= self._n_active_persons and not person.detected:
                person.detected = True
                person.hidden = True     # hide from render
                pre_found += 1
        self._pre_detected_count = pre_found
        self._active_total = max(
            1, len(self._env.persons) - self._pre_detected_count
        )
        self._env.total_found += pre_found

    @staticmethod
    def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (_TWO_PI) - np.pi

    def _update_sensor_coverage(self) -> int:
        centers = self._coverage_cell_centers
        observed_new = np.zeros(COVERAGE_DIM, dtype=bool)
        uncovered = self._coverage_grid < 0.5

        if not np.any(uncovered):
            return 0

        for robot in self._env.robots:
            rx = float(robot.x)
            ry = float(robot.y)
            dx = centers[:, 0] - rx
            dy = centers[:, 1] - ry
            candidates = uncovered.copy()
            d2 = dx * dx + dy * dy
            candidates &= (d2 <= _FOV_RANGE_SQ)
            if not np.any(candidates):
                continue

            sensor_hdg = float(robot.theta + robot.sensor_off)
            cand_idx = np.flatnonzero(candidates)
            angles = np.arctan2(dy[cand_idx], dx[cand_idx])
            rel = (angles - sensor_hdg + np.pi) % _TWO_PI - np.pi
            in_cone = np.abs(rel) <= _HALF_FOV
            observed_new[cand_idx[in_cone]] = True

        new_count = int(np.count_nonzero(observed_new))
        if new_count > 0:
            self._coverage_grid[observed_new] = 1.0
        return new_count

    def _nearest_uncovered_info(self, pos: np.ndarray) -> tuple[float, float]:
        uncovered_mask = self._coverage_grid < 0.5
        if not np.any(uncovered_mask):
            return 0.0, 0.0

        uncovered_centers = self._coverage_cell_centers[uncovered_mask]
        dx = uncovered_centers[:, 0] - float(pos[0])
        dy = uncovered_centers[:, 1] - float(pos[1])
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        dist = float(np.sqrt(d2[idx]))

        bearing = float(np.arctan2(dy[idx], dx[idx]))
        heading = float(self._env.robots[0].theta)
        bearing_err = (bearing - heading + np.pi) % _TWO_PI - np.pi

        return dist, float(bearing_err)

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = raw_obs.copy()
        det_vals = obs[self._person_det_indices]
        undetected_mask = np.repeat(det_vals < 0.5, 2)
        obs[self._person_xy_indices[undetected_mask]] = 0.0

        coverage_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
        time_remaining = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
        stagnation_frac = min(
            self._steps_since_new_cell / (STAGNATION_GRACE + STAGNATION_RAMP),
            1.0,
        )
        extra = np.array(
            [coverage_frac, time_remaining, stagnation_frac], dtype=np.float32
        )
        return np.concatenate([obs, self._coverage_grid, extra], dtype=np.float32)

    def _min_known_clearance_vec(self, pos: np.ndarray, known_obs: list) -> float:
        if not known_obs:
            return 1e6
        obs_arr = np.array(known_obs, dtype=np.float64)
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
        clearance: float, safe_clearance: float, alpha: float, max_penalty: float,
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

    def _reset_reward_components(self) -> None:
        self._reward_components = {
            "person_found": 0.0, "collision": 0.0, "formation": 0.0,
            "formation_penalty": 0.0, "follower_approach": 0.0,
            "coverage_new": 0.0, "stagnation": 0.0, "explore_dist": 0.0,
            "explore_heading": 0.0, "forward_vel": 0.0, "spin": 0.0,
            "heading_jitter": 0.0, "prox_obstacle": 0.0, "prox_wall": 0.0,
            "time_penalty": 0.0, "early_completion": 0.0,
        }

    def _shape_reward(
        self, info: dict, raw_obs: np.ndarray, action: np.ndarray,
        new_coverage_cells: int, min_obs_clearance: float, min_wall_clearance: float,
        dist_to_uncovered: float, bearing_to_uncovered: float, all_found: bool,
    ) -> float:
        rc = self._reward_components
        reward = 0.0

        current_found = info["found"]
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            reward_per_person = TOTAL_RESCUE_POOL / max(float(self._active_total), 1.0)
            r = reward_per_person * new_detections
            rc["person_found"] += r
            reward += r
        self._prev_found = current_found

        if info["collisions"] > self._prev_collisions:
            rc["collision"] += R_COLLISION
            reward += R_COLLISION
        self._prev_collisions = info["collisions"]

        if info["form_dev_mean"] < 1.5:
            rc["formation"] += R_FORMATION_BONUS
            reward += R_FORMATION_BONUS
        elif info["form_dev_mean"] > 4.5:
            # Scaled penalty: worse formation = bigger penalty, capped at 3x base
            excess = min((info["form_dev_mean"] - 4.5) / 5.0, 1.0)
            r = R_FORMATION_PENALTY * (1.0 + 2.0 * excess)
            rc["formation_penalty"] += r
            reward += r

        # Reward followers for moving toward their formation slots
        leader_pose = self._env.robots[0].pose
        f1_tgt = body_to_world(FORM_OFFSET[1], leader_pose)
        f2_tgt = body_to_world(FORM_OFFSET[2], leader_pose)
        f1_dist = float(np.linalg.norm(self._env.robots[1].pos - f1_tgt))
        f2_dist = float(np.linalg.norm(self._env.robots[2].pos - f2_tgt))

        # Positive when closing distance, negative when drifting
        f1_delta = self._prev_f1_form_dist - f1_dist
        f2_delta = self._prev_f2_form_dist - f2_dist
        approach_r = R_FOLLOWER_APPROACH * (f1_delta + f2_delta)
        approach_r = min(max(approach_r, -0.5), 0.5)  # clamp per-step
        rc["follower_approach"] += approach_r
        reward += approach_r
        self._prev_f1_form_dist = f1_dist
        self._prev_f2_form_dist = f2_dist

        if new_coverage_cells > 0:
            cov_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
            multiplier = 1.0 + cov_frac ** R_COVERAGE_PROGRESS_EXP
            r = R_COVERAGE_NEW * float(new_coverage_cells) * multiplier
            rc["coverage_new"] += r
            reward += r

        if self._steps_since_new_cell > STAGNATION_GRACE:
            overshoot = self._steps_since_new_cell - STAGNATION_GRACE
            ramp_frac = min(overshoot / STAGNATION_RAMP, 1.0)
            r = R_STAGNATION_BASE + (R_STAGNATION_MAX - R_STAGNATION_BASE) * ramp_frac
            rc["stagnation"] += r
            reward += r

        if self._prev_dist_to_uncovered > 0 and dist_to_uncovered > 0:
            delta = (self._prev_dist_to_uncovered - dist_to_uncovered) / _MAP_DIAG
            r = R_EXPLORATION_GRADIENT * min(max(delta * 10.0, -1.0), 1.0)
            rc["explore_dist"] += r
            reward += r

        if dist_to_uncovered > 0:
            alignment = float(np.cos(bearing_to_uncovered))
            r = R_HEADING_TO_UNCOVERED * alignment
            rc["explore_heading"] += r
            reward += r

        # Reward forward velocity using actual post-smoothing velocities
        av = getattr(self, '_actual_vels', None)
        if av is not None:
            v_l_actual, w_l_actual, v_f1_actual, w_f1_actual, v_f2_actual, w_f2_actual = av
            v_fraction_avg = (v_l_actual + v_f1_actual + v_f2_actual) / (3.0 * V_MAX) if V_MAX > 0 else 0.0
            r = R_FORWARD_VELOCITY * v_fraction_avg
            rc["forward_vel"] += r
            reward += r

            # Penalize excessive spinning using actual angular velocities
            omega_avg = (abs(w_l_actual) + abs(w_f1_actual) + abs(w_f2_actual)) / (3.0 * OMEGA_MAX) if OMEGA_MAX > 0 else 0.0
            r = R_SPIN_PENALTY_SCALE * omega_avg
            rc["spin"] += r
            reward += r
        else:
            # Fallback to raw actions if actual vels not yet available
            v_l_frac = (float(action[0]) + 1.0) * 0.5
            v_f1_frac = (float(action[2]) + 1.0) * 0.5
            v_f2_frac = (float(action[4]) + 1.0) * 0.5
            v_fraction_avg = (v_l_frac + v_f1_frac + v_f2_frac) / 3.0
            r = R_FORWARD_VELOCITY * v_fraction_avg
            rc["forward_vel"] += r
            reward += r

            omega_l_frac = abs(float(action[1]))
            omega_f1_frac = abs(float(action[3]))
            omega_f2_frac = abs(float(action[5]))
            omega_fraction_avg = (omega_l_frac + omega_f1_frac + omega_f2_frac) / 3.0
            r = R_SPIN_PENALTY_SCALE * omega_fraction_avg
            rc["spin"] += r
            reward += r

        heading = float(self._env.robots[0].theta)
        delta_heading = abs((heading - self._prev_heading + np.pi) % _TWO_PI - np.pi)
        r = R_HEADING_CHANGE_SCALE * (delta_heading / np.pi)
        rc["heading_jitter"] += r
        reward += r
        self._prev_heading = heading

        r_obs = self._exp_proximity_penalty(
            min_obs_clearance, SAFETY_SLOW_CLEARANCE, EXP_OBS_ALPHA, R_NEAR_OBS_EXP_MAX,
        )
        r_wall = self._exp_proximity_penalty(
            min_wall_clearance, WALL_SAFE_CLEARANCE, EXP_WALL_ALPHA, R_NEAR_WALL_EXP_MAX,
        )
        rc["prox_obstacle"] += r_obs
        rc["prox_wall"] += r_wall
        reward += r_obs + r_wall

        rc["time_penalty"] += R_TIME_PENALTY
        reward += R_TIME_PENALTY

        if all_found:
            frac_remaining = 1.0 - (self._step_count / MAX_EPISODE_STEPS)
            r = R_EARLY_COMPLETION * frac_remaining
            rc["early_completion"] += r
            reward += r

        rc["unclipped_total"] = rc.get("unclipped_total", 0.0) + reward
        reward = min(max(reward, REWARD_CLIP_LOW), REWARD_CLIP_HIGH)
        return reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._env.seed = seed
        else:
            self._env.seed = self._base_seed + self._episode_count
        self._episode_count += 1

        if self._random_difficulty:
            chosen_persons = int(np.random.choice(RANDOM_STAGE4_PERSONS, p=self._stage4_weights))
            obs_map = {3: 2, 5: 4, 7: 6, 10: 7}
            self.set_n_active_persons(chosen_persons)
            self.set_n_active_obstacles(obs_map[chosen_persons])

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
        self._prev_v_f1 = 0.0
        self._prev_w_f1 = 0.0
        self._prev_v_f2 = 0.0
        self._prev_w_f2 = 0.0
        self._prev_heading = float(self._env.robots[0].theta)
        dist, _ = self._nearest_uncovered_info(self._env.robots[0].pos)
        self._prev_dist_to_uncovered = dist
        # Init follower formation tracking
        leader = self._env.robots[0]
        f1_tgt = body_to_world(FORM_OFFSET[1], leader.pose)
        f2_tgt = body_to_world(FORM_OFFSET[2], leader.pose)
        self._prev_f1_form_dist = float(np.linalg.norm(self._env.robots[1].pos - f1_tgt))
        self._prev_f2_form_dist = float(np.linalg.norm(self._env.robots[2].pos - f2_tgt))
        self._reset_reward_components()
        self._clipped_ep_reward = 0.0

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(action, -1.0, 1.0)
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()

        # 6DOF Command mapping
        v_l  = float((a[0] + 1.0) * 0.5 * V_MAX)
        w_l  = float(a[1] * OMEGA_MAX)
        v_f1 = float((a[2] + 1.0) * 0.5 * V_MAX)
        w_f1 = float(a[3] * OMEGA_MAX)
        v_f2 = float((a[4] + 1.0) * 0.5 * V_MAX)
        w_f2 = float(a[5] * OMEGA_MAX)

        # Apply leader safety limits
        min_obs_clearance = self._min_known_clearance_vec(leader.pos, known_obs)
        min_wall_clearance = self._min_wall_clearance(leader.pos)
        min_clearance = min(min_obs_clearance, min_wall_clearance)
        if min_clearance < SAFETY_STOP_CLEARANCE:
            v_l = min(v_l, 0.15 * V_MAX)
        elif min_clearance < SAFETY_SLOW_CLEARANCE:
            v_l = min(v_l, 0.50 * V_MAX)

        # Apply follower-1 safety limits
        f1_obs_cl = self._min_known_clearance_vec(f1.pos, known_obs)
        f1_wall_cl = self._min_wall_clearance(f1.pos)
        f1_min_cl = min(f1_obs_cl, f1_wall_cl)
        if f1_min_cl < SAFETY_STOP_CLEARANCE:
            v_f1 = min(v_f1, 0.15 * V_MAX)
        elif f1_min_cl < SAFETY_SLOW_CLEARANCE:
            v_f1 = min(v_f1, 0.50 * V_MAX)

        # Apply follower-2 safety limits
        f2_obs_cl = self._min_known_clearance_vec(f2.pos, known_obs)
        f2_wall_cl = self._min_wall_clearance(f2.pos)
        f2_min_cl = min(f2_obs_cl, f2_wall_cl)
        if f2_min_cl < SAFETY_STOP_CLEARANCE:
            v_f2 = min(v_f2, 0.15 * V_MAX)
        elif f2_min_cl < SAFETY_SLOW_CLEARANCE:
            v_f2 = min(v_f2, 0.50 * V_MAX)

        # Smooth leader commands
        alpha = COMMAND_SMOOTH_ALPHA
        one_minus_alpha = 1.0 - alpha
        v_l = alpha * self._prev_v_l + one_minus_alpha * v_l
        w_l = alpha * self._prev_w_l + one_minus_alpha * w_l
        v_l = min(max(v_l, 0.0), V_MAX)
        w_l = min(max(w_l, -OMEGA_MAX), OMEGA_MAX)
        self._prev_v_l = v_l
        self._prev_w_l = w_l

        # Smooth follower-1 commands
        v_f1 = alpha * self._prev_v_f1 + one_minus_alpha * v_f1
        w_f1 = alpha * self._prev_w_f1 + one_minus_alpha * w_f1
        v_f1 = min(max(v_f1, 0.0), V_MAX)
        w_f1 = min(max(w_f1, -OMEGA_MAX), OMEGA_MAX)
        self._prev_v_f1 = v_f1
        self._prev_w_f1 = w_f1

        # Smooth follower-2 commands
        v_f2 = alpha * self._prev_v_f2 + one_minus_alpha * v_f2
        w_f2 = alpha * self._prev_w_f2 + one_minus_alpha * w_f2
        v_f2 = min(max(v_f2, 0.0), V_MAX)
        w_f2 = min(max(w_f2, -OMEGA_MAX), OMEGA_MAX)
        self._prev_v_f2 = v_f2
        self._prev_w_f2 = w_f2

        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1
        # Store actual velocities for reward calculation (post-safety, post-smoothing)
        self._actual_vels = (v_l, w_l, v_f1, w_f1, v_f2, w_f2)

        new_cells = self._update_sensor_coverage()

        if new_cells > 0:
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1

        dist_to_uncov, bearing_to_uncov = self._nearest_uncovered_info(leader.pos)

        collided_now = info["collisions"] > self._prev_collisions
        active_found = int(np.clip(
            info["found"] - self._pre_detected_count, 0, self._active_total,
        ))
        all_found = active_found >= self._active_total

        obs = self._build_obs(raw_obs)
        # Proximity penalty uses worst clearance across ALL robots (not just leader)
        worst_obs_clearance = min(min_obs_clearance, f1_obs_cl, f2_obs_cl)
        worst_wall_clearance = min(min_wall_clearance, f1_wall_cl, f2_wall_cl)
        reward = self._shape_reward(
            info, raw_obs, a, new_cells,
            worst_obs_clearance, worst_wall_clearance,
            dist_to_uncov, bearing_to_uncov, all_found,
        )
        self._clipped_ep_reward += reward

        self._prev_dist_to_uncovered = dist_to_uncov

        terminated = all_found
        if TERMINATE_ON_COLLISION and collided_now:
            terminated = True

        # End hopelessly stuck episodes early to avoid long dead rollouts.
        if self._steps_since_new_cell >= MAX_STAGNATION_STEPS:
            truncated = True
        else:
            truncated = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

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

        if terminated or truncated:
            info["reward_components"] = dict(self._reward_components)
            info["clipped_ep_reward"] = self._clipped_ep_reward

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor("#12121f")
            plt.ion()

        leader = self._env.robots[0]
        f1_tgt = body_to_world(FORM_OFFSET[1], leader.pose)
        f2_tgt = body_to_world(FORM_OFFSET[2], leader.pose)

        # hidden flag on pre-detected persons means render already skips them
        self._env.render(self._ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)

        # Coverage overlay: single imshow instead of per-cell Rectangles
        cov_2d = self._coverage_grid.reshape(GRID_N, GRID_N)
        rgba = np.zeros((GRID_N, GRID_N, 4), dtype=np.float32)
        rgba[cov_2d > 0.5, :] = [0.0, 0.9, 0.46, 0.08]  # #00e676 at 8% alpha
        self._ax.imshow(
            rgba.transpose(1, 0, 2),
            extent=[0, MAP_W, 0, MAP_H],
            origin="lower", interpolation="nearest", zorder=1,
        )

        stag = self._steps_since_new_cell
        self._fig.suptitle(
            "SAR Swarm [RL 6DOF] "
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
def _load_curriculum_state(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def make_env(rank: int, base_seed: int = SEED):
    def _init() -> gym.Env:
        env = SARGymnasiumWrapper(seed=base_seed + rank * 1000)
        return Monitor(env)
    return _init


# ===========================================================================
# SECTION 4 | CALLBACKS
# ===========================================================================
class MasteryStatusCallback(BaseCallback):
    def __init__(self, state_path: str, eval_env=None, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._state_path = state_path
        self._eval_env = eval_env  # sync eval difficulty with curriculum
        self.stage_start_time = time.time()
        self.training_start_time = time.time()
        self.last_print_step = 0
        self._last_status_time = 0.0
        self.current_stage_idx = -1
        self._stage_start_step = 0
        self._stage_episode_count = 0
        self._stage_collision_sum = 0.0
        self._stage_strict_success_sum = 0.0
        self._recent_strict: list[float] = []
        self._recovery_mode = False

    def _reset_stage_accumulators(self) -> None:
        self._stage_episode_count = 0
        self._stage_collision_sum = 0.0
        self._stage_strict_success_sum = 0.0

    def _consume_done_infos(self) -> None:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            collisions = float(info.get("collisions", 0.0))
            active_total = max(float(info.get("active_total", info.get("total", 1.0))), 1.0)
            active_found = float(np.clip(info.get("active_found", info.get("found", 0.0)), 0.0, active_total))
            strict_success = 1.0 if (active_found >= active_total and collisions <= 0.0) else 0.0
            self._stage_episode_count += 1
            self._stage_collision_sum += collisions
            self._stage_strict_success_sum += strict_success

            self._recent_strict.append(strict_success)
            max_n = max(int(NO_REGRESSION_WINDOW_EPISODES), 1)
            if len(self._recent_strict) > max_n:
                self._recent_strict = self._recent_strict[-max_n:]

    def _apply_stage4_sampling_policy(self, force: bool = False) -> None:
        if self.current_stage_idx != len(CURRICULUM_STAGES) - 1:
            return
        if len(self._recent_strict) < max(int(NO_REGRESSION_WINDOW_EPISODES // 2), 1):
            if force:
                self.training_env.env_method("set_stage4_sampling_weights", RANDOM_STAGE4_WEIGHTS.tolist())
            return

        rolling_strict = float(np.mean(self._recent_strict))
        if (not self._recovery_mode) and rolling_strict < float(NO_REGRESSION_ENTER_FLOOR):
            self._recovery_mode = True
            self.training_env.env_method("set_stage4_sampling_weights", RANDOM_STAGE4_RECOVERY_WEIGHTS.tolist())
            print(
                f"\n>>> No-regression gate ON | rolling strict={rolling_strict * 100.0:.1f}% "
                f"(< {NO_REGRESSION_ENTER_FLOOR * 100.0:.1f}%) | biasing easier mastery mix"
            )
        elif self._recovery_mode and rolling_strict >= float(NO_REGRESSION_EXIT_FLOOR):
            self._recovery_mode = False
            self.training_env.env_method("set_stage4_sampling_weights", RANDOM_STAGE4_WEIGHTS.tolist())
            print(
                f"\n>>> No-regression gate OFF | rolling strict={rolling_strict * 100.0:.1f}% "
                f"(>= {NO_REGRESSION_EXIT_FLOOR * 100.0:.1f}%) | restoring hard mastery mix"
            )

    def _print_stage_summary(self, stage_idx: int) -> None:
        if self._stage_episode_count <= 0:
            return
        stage_minutes = (time.time() - self.stage_start_time) / 60.0
        avg_collisions = self._stage_collision_sum / self._stage_episode_count
        strict_success_pct = 100.0 * self._stage_strict_success_sum / self._stage_episode_count
        print(
            f"\n>>> Stage {stage_idx + 1} Summary | "
            f"Episodes: {self._stage_episode_count} | "
            f"Avg collisions: {avg_collisions:.2f} | "
            f"Strict success: {strict_success_pct:.1f}% | "
            f"Stage time: {stage_minutes:.1f}m"
        )

    def _save_state(self, n_persons: int, n_obstacles: int) -> None:
        payload = {
            "stage_idx": int(self.current_stage_idx),
            "stage_start_step": int(self._stage_start_step),
            "stage_elapsed_sec": float(max(time.time() - self.stage_start_time, 0.0)),
            "active_persons": int(n_persons),
            "active_obstacles": int(n_obstacles),
            "random_difficulty": bool(self.current_stage_idx == len(CURRICULUM_STAGES) - 1),
            "num_timesteps": int(self.num_timesteps),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _on_step(self) -> bool:
        self._consume_done_infos()

        target_stage = 0
        for i, (threshold, _, _) in enumerate(CURRICULUM_STAGES):
            if self.num_timesteps >= threshold:
                target_stage = i

        if target_stage != self.current_stage_idx:
            if self.current_stage_idx >= 0:
                self._print_stage_summary(self.current_stage_idx)
                self._reset_stage_accumulators()

            self.current_stage_idx = target_stage
            self.stage_start_time = time.time()
            self._stage_start_step = self.num_timesteps
            _, n_persons, n_obstacles = CURRICULUM_STAGES[target_stage]

            if target_stage == len(CURRICULUM_STAGES) - 1:
                self.training_env.env_method("enable_random_difficulty", True)
                self.training_env.env_method("set_stage4_sampling_weights", RANDOM_STAGE4_WEIGHTS.tolist())
                self._recovery_mode = False
                self._recent_strict.clear()
            else:
                self.training_env.env_method("enable_random_difficulty", False)
                self.training_env.env_method("set_n_active_persons", n_persons)
                self.training_env.env_method("set_n_active_obstacles", n_obstacles)

            # Keep eval complexity aligned with the current curriculum stage.
            if self._eval_env is not None:
                self._eval_env.env_method("set_n_active_persons", n_persons)
                self._eval_env.env_method("set_n_active_obstacles", n_obstacles)

            stars = "*" * (target_stage + 1)
            print(
                f"\n{stars} Curriculum stage {target_stage + 1}"
                f"/{len(CURRICULUM_STAGES)} — "
                f"{n_persons} persons, {n_obstacles} obstacles "
                f"(t={self.num_timesteps:,}) {stars}\n"
            )
            self._save_state(n_persons=n_persons, n_obstacles=n_obstacles)

        # Periodic state save (no custom status line — tqdm handles the timer)
        now = time.time()
        if (now - self._last_status_time) >= 30.0:
            self._last_status_time = now
            _, n_persons, n_obstacles = CURRICULUM_STAGES[self.current_stage_idx]
            self._save_state(n_persons=n_persons, n_obstacles=n_obstacles)

        self._apply_stage4_sampling_policy()

        return True


class BestModelSweepCallback(BaseCallback):
    def __init__(
        self,
        model_prefix: str,
        device: str,
        status_callback=None,
        n_episodes: int = BEST_SWEEP_EPISODES,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._model_prefix = model_prefix
        self._device = device
        self._status_callback = status_callback  # to read current_stage_idx for stage-aware difficulty
        self._n_episodes = max(int(n_episodes), 1)
        self._last_mtime: Optional[float] = None
        self._last_report_step: int = -1

    def _run_sweep(self) -> None:
        model = RecurrentPPO.load(self._model_prefix, device=self._device)
        rewards: list[float] = []
        collisions: list[float] = []
        strict_success: list[float] = []

        # Get current stage difficulty (stage-aware evaluation like main EvalCallback)
        stage_idx = 0
        if self._status_callback is not None and self._status_callback.current_stage_idx >= 0:
            stage_idx = self._status_callback.current_stage_idx
        _, n_persons, n_obstacles = CURRICULUM_STAGES[stage_idx]

        for ep in range(self._n_episodes):
            env = SARGymnasiumWrapper(seed=SEED + 200_000 + ep)
            env.set_n_active_persons(n_persons)
            env.set_n_active_obstacles(n_obstacles)
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            lstm_states = None
            episode_starts = np.array([True], dtype=bool)
            info = {}

            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                episode_starts = np.array([done], dtype=bool)

            active_total = max(float(info.get("active_total", info.get("total", 1.0))), 1.0)
            active_found = float(np.clip(info.get("active_found", info.get("found", 0.0)), 0.0, active_total))
            coll = float(info.get("collisions", 0.0))
            rewards.append(ep_reward)
            collisions.append(coll)
            strict_success.append(1.0 if (active_found >= active_total and coll <= 0.0) else 0.0)
            env.close()

        print(
            f"\n>>> New best checkpoint sweep ({self._n_episodes} eps) | "
            f"Reward: {float(np.mean(rewards)):.1f} | "
            f"Collisions: {float(np.mean(collisions)):.2f} | "
            f"Strict success: {100.0 * float(np.mean(strict_success)):.1f}%"
        )

    def _on_step(self) -> bool:
        model_zip = f"{self._model_prefix}.zip"
        if not os.path.exists(model_zip):
            return True

        mtime = os.path.getmtime(model_zip)
        if self._last_mtime is None:
            self._last_mtime = mtime
            return True

        if mtime > self._last_mtime and self.num_timesteps != self._last_report_step:
            self._last_mtime = mtime
            self._last_report_step = self.num_timesteps
            self._run_sweep()
        return True


class InstabilityGuardCallback(BaseCallback):
    def __init__(
        self,
        checkpoint_dir: str,
        vecnorm_path: str,
        check_every_steps: int = 2000,
        param_abs_limit: float = 1e6,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._checkpoint_dir = checkpoint_dir
        self._vecnorm_path = vecnorm_path
        self._check_every_steps = max(int(check_every_steps), 1)
        self._param_abs_limit = float(param_abs_limit)
        self._last_check = 0

    def _emergency_save_and_stop(self, reason: str) -> bool:
        ts = int(time.time())
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        emergency_prefix = os.path.join(self._checkpoint_dir, f"emergency_{self.num_timesteps}_{ts}")
        print(f"\n[!] Instability detected: {reason}")
        print(f"[!] Saving emergency checkpoint -> {emergency_prefix}.zip")
        try:
            self.model.save(emergency_prefix)
        except Exception as exc:
            print(f"[!] Emergency model save failed: {exc}")
        try:
            self.training_env.save(self._vecnorm_path)
        except Exception as exc:
            print(f"[!] Emergency VecNormalize save failed: {exc}")
        return False

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self._check_every_steps:
            return True
        self._last_check = self.num_timesteps

        max_abs = 0.0
        for p in self.model.policy.parameters():
            if not torch.isfinite(p).all():
                return self._emergency_save_and_stop("non-finite policy parameters")
            local_max = float(torch.max(torch.abs(p)).item())
            if local_max > max_abs:
                max_abs = local_max
        if max_abs > self._param_abs_limit:
            return self._emergency_save_and_stop(f"parameter explosion (max |w|={max_abs:.3e})")

        infos = self.locals.get("infos", [])
        for info in infos:
            cov = info.get("coverage_frac", 0.0)
            if isinstance(cov, (float, int)) and not np.isfinite(float(cov)):
                return self._emergency_save_and_stop("non-finite coverage metric")
        return True


class SARMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            if not done or "found" not in info:
                continue

            active_total = max(info.get("active_total", info.get("total", 1)), 1)
            active_found = float(np.clip(info.get("active_found", info.get("found", 0)), 0, active_total))
            stage = 1
            for i, (threshold, _, _) in enumerate(CURRICULUM_STAGES):
                if self.num_timesteps >= threshold:
                    stage = i + 1

            self.logger.record("sar/persons_found", active_found)
            self.logger.record("sar/persons_total", float(active_total))
            self.logger.record("sar/find_rate", active_found / active_total)
            self.logger.record("sar/collisions", float(info["collisions"]))
            self.logger.record("sar/form_dev_mean", info["form_dev_mean"])
            self.logger.record("sar/coverage_frac", info.get("coverage_frac", 0.0))
            self.logger.record("sar/curriculum_stage", float(stage))
            self.logger.record("sar/steps_since_new_cell", float(info.get("steps_since_new_cell", 0)))

            rc = info.get("reward_components", {})
            for name, value in rc.items():
                self.logger.record(f"reward/{name}", value)
            if "unclipped_total" in rc:
                self.logger.record("sar/unclipped_ep_reward", rc["unclipped_total"])
            if "clipped_ep_reward" in info:
                self.logger.record("sar/clipped_ep_reward", info["clipped_ep_reward"])

        return True


# ===========================================================================
# SECTION 5 | SURGICAL WEIGHT TRANSFER
# ===========================================================================
def transfer_v9_weights(model: RecurrentPPO, v9_path: str, device: str):
    """
    Loads v9 checkpoint, copies LSTM/Features/Value net weights, 
    but skips action_net and log_std because the output dimension changed from 2 to 6.
    """
    print(f"\n[!] Surgically transferring weights from: {v9_path}.zip")
    if not os.path.exists(f"{v9_path}.zip"):
        raise FileNotFoundError(f"Could not find {v9_path}.zip for weight transfer!")

    pretrained_model = RecurrentPPO.load(v9_path, device=device)
    pretrained_dict = pretrained_model.policy.state_dict()
    new_model_dict = model.policy.state_dict()

    transferred = 0
    skipped = 0

    for key, param in pretrained_dict.items():
        if "action_net" in key or "log_std" in key:
            print(f"    Skipping: {key} (shape mismatch due to 6DOF)")
            skipped += 1
            continue

        if key in new_model_dict and new_model_dict[key].shape == param.shape:
            new_model_dict[key].copy_(param)
            transferred += 1
        else:
            reason = "not in new model" if key not in new_model_dict else (
                f"shape {param.shape} vs {new_model_dict[key].shape}"
            )
            print(f"    Skipping: {key} ({reason})")
            skipped += 1

    model.policy.load_state_dict(new_model_dict)
    print(f"    Weight transfer complete: {transferred} layers copied, {skipped} layers skipped.\n")


# ===========================================================================
# SECTION 6 | TRAINING FUNCTION
# ===========================================================================
def train(device: str = "auto", resume: bool = False) -> None:
    print("=" * 65)
    print("  SAR SWARM — RecurrentPPO + LSTM (v10 — 6DOF Full Control)")
    print("=" * 65)
    print(
        f"  Architecture : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']}) "
        f"+ dense {POLICY_KWARGS['net_arch']}"
    )
    print("  Action space : 6-D (Leader + Follower 1 + Follower 2)")
    print(
        f"  Obs dim      : {SARGymnasiumWrapper.OBS_DIM} "
        f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM}"
        f" + extra {EXTRA_OBS_DIM})"
    )
    print(f"  Workers      : {N_ENVS}")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Device       : {device}")
    print("  Curriculum:")
    for i, (thr, n, o) in enumerate(CURRICULUM_STAGES):
        print(f"    Stage {i + 1}: {n:>2} persons, {o} obstacles (from step {thr:>9,})")
    print("=" * 65)

    print("\n[1/5] Checking environment wrapper...")
    _check = SARGymnasiumWrapper(seed=SEED)
    check_env(_check, warn=True)
    _check.close()
    print("      OK — Wrapper check passed.\n")

    print("[2/5] Spawning training environments...")
    env_fns = [make_env(rank=i, base_seed=SEED) for i in range(N_ENVS)]
    train_env = SubprocVecEnv(env_fns, start_method=_MP_START_METHOD)
    train_env = VecMonitor(train_env)

    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    train_env.env_method("set_n_active_persons", CURRICULUM_STAGES[0][1])
    train_env.env_method("set_n_active_obstacles", CURRICULUM_STAGES[0][2])
    print(f"      OK — {N_ENVS} workers ready (VecNormalize active).\n")

    print("[3/5] Building evaluation environment (stage-aware difficulty, fixed seed)...")
    eval_base = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)),
        n_envs=1,
    )
    eval_base = VecMonitor(eval_base)
    eval_env = VecNormalize(
        eval_base,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    eval_env.env_method("set_n_active_persons", CURRICULUM_STAGES[0][1])
    eval_env.env_method("set_n_active_obstacles", CURRICULUM_STAGES[0][2])
    print("      OK — Eval environment ready (stage 1 difficulty).\n")

    print("[4/5] Instantiating RecurrentPPO model for 6DOF...")
    best_6dof_path = "checkpoints_6dof/best_model"

    if resume and os.path.exists(f"{best_6dof_path}.zip"):
        print(f"      Resuming directly from best 6DOF checkpoint: {best_6dof_path}.zip")
        model = RecurrentPPO.load(
            best_6dof_path, env=train_env, device=device,
            tensorboard_log=TENSORBOARD_LOG,
        )
    else:
        print("      Building new 6DOF model and transferring v9 brain...")
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
            target_kl=0.05,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=TENSORBOARD_LOG,
            verbose=1,
            seed=SEED,
            device=device,
        )

        # Run the weight transfer from v9 when starting fresh.
        transfer_v9_weights(model, V9_BEST_MODEL_PATH, device)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"      OK — RecurrentPPO built. Parameters: {n_params:,}\n")

    print("[5/5] Configuring callbacks...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    if not resume:
        # Fresh run: reset so first eval establishes a new best.
        eval_callback.best_mean_reward = -float("inf")
    else:
        # Resume: restore previous best from eval log so we don't overwrite
        # a better checkpoint with a worse one.
        _eval_log = os.path.join(LOG_DIR, "evaluations.npz")
        if os.path.exists(_eval_log):
            _eval_data = np.load(_eval_log)
            _prev_means = np.mean(_eval_data["results"], axis=1)
            _prev_best = float(np.max(_prev_means))
            eval_callback.best_mean_reward = _prev_best
            print(f"      Restored previous best eval reward: {_prev_best:.1f}")
        else:
            eval_callback.best_mean_reward = -float("inf")

    status_callback = MasteryStatusCallback(state_path=CURRICULUM_STATE_PATH, eval_env=eval_env, verbose=1)
    sweep_callback = BestModelSweepCallback(
        model_prefix=os.path.join(CHECKPOINT_DIR, "best_model"),
        device=device,
        status_callback=status_callback,
        n_episodes=BEST_SWEEP_EPISODES,
        verbose=1,
    )
    guard_callback = InstabilityGuardCallback(
        checkpoint_dir=CHECKPOINT_DIR,
        vecnorm_path=VECNORM_PATH,
        check_every_steps=2000,
        param_abs_limit=1e6,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_sar_6dof",
        verbose=1,
    )

    if resume:
        state = _load_curriculum_state(CURRICULUM_STATE_PATH)
        if state is not None:
            restored_stage = int(state.get("stage_idx", -1))
            restored_random = bool(state.get("random_difficulty", False))
            restored_people = int(state.get("active_persons", CURRICULUM_STAGES[0][1]))
            restored_obs = int(state.get("active_obstacles", CURRICULUM_STAGES[0][2]))
            if restored_random:
                train_env.env_method("enable_random_difficulty", True)
            else:
                train_env.env_method("enable_random_difficulty", False)
                train_env.env_method("set_n_active_persons", restored_people)
                train_env.env_method("set_n_active_obstacles", restored_obs)

            # Keep eval env in sync after resume as well.
            eval_env.env_method("set_n_active_persons", restored_people)
            eval_env.env_method("set_n_active_obstacles", restored_obs)

            status_callback.current_stage_idx = restored_stage
            status_callback._stage_start_step = int(state.get("stage_start_step", model.num_timesteps))
            status_callback.stage_start_time = time.time() - float(state.get("stage_elapsed_sec", 0.0))
            print(
                "      Restored curriculum state: "
                f"stage={restored_stage + 1}, random={restored_random}, "
                f"persons={restored_people}, obstacles={restored_obs}"
            )

    callbacks = [
        status_callback,
        SARMetricsCallback(verbose=0),
        eval_callback,
        sweep_callback,
        guard_callback,
        checkpoint_callback,
    ]
    print(f"      OK — {len(callbacks)} callbacks attached.\n")

    print(f"Starting 6DOF training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  TensorBoard: tensorboard --logdir ppo_sar_tensorboard_6dof\n")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not resume,
            tb_log_name="RecurrentPPO_SAR_6DOF",
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n[!] Training interrupted: {e}")
        print("[!] Emergency save...")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    model.save(MODEL_SAVE_PATH)
    train_env.save(VECNORM_PATH)
    print(f"OK — Model saved to '{MODEL_SAVE_PATH}.zip'")
    print(f"OK — VecNormalize saved to '{VECNORM_PATH}'")
    train_env.close()
    eval_env.close()


# ===========================================================================
# SECTION 8 | ENTRY POINT
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RecurrentPPO (LSTM) 6DOF training for the SAR Swarm."
    )
    parser.add_argument(
        "--mode", choices=["train", "check"],
        default="train", help="train (default) | check",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"],
        default="auto", help="Torch device (default: auto → uses CUDA if available)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the best 6DOF checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _enable_live_train_logging("train_log_6dof.txt")
    args = parse_args()

    if args.mode == "check":
        print("Running SB3 environment checker...")
        env = SARGymnasiumWrapper(seed=SEED)
        check_env(env, warn=True)
        env.close()
        print("OK — check_env() passed.")

    elif args.mode == "train":
        train(device=args.device, resume=args.resume)