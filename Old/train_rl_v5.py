"""
train_rl.py (v5 — Reward System Overhaul & Code Hardening)

Changes from v4:
  ✓ REWARD REBALANCE: All rewards compressed to [-5, +10] effective range per step.
    No single event dominates the value function.
  ✓ ANTI-SPINNING: Explicit angular-velocity penalty + heading-change penalty.
  ✓ EXPLORATION GRADIENT: Distance-to-nearest-uncovered-cell reward gives the agent
    a smooth signal pointing toward unexplored territory.
  ✓ PROGRESSIVE COVERAGE: Diminishing returns on easy early cells; bonus multiplier
    as coverage fraction grows (harder cells are worth more).
  ✓ COLLISION: Reduced penalty since TERMINATE_ON_COLLISION already ends the episode.
    Double-punishing destabilises the value function.
  ✓ REMOVED DEAD CODE: R_FORWARD=0, R_NO_COLLISION_STEP=0 deleted.
  ✓ SYNTAX FIXES: Corrected dunder imports/names, added missing type hints.
  ✓ BATCH SIZE: Fixed for RecurrentPPO (must divide N_STEPS, not total rollout).
  ✓ ENTROPY SCHEDULE: Cosine-decayed entropy so early exploration is aggressive
    but late-stage policy sharpens.
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

# Prevent PyTorch from using all CPU cores and fighting the Gym environments
torch.set_num_threads(1)

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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
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
    """Cosine-annealed entropy coefficient.

    Starts high (aggressive exploration) and decays to `final` for
    sharper late-stage policies. progress_remaining goes 1 → 0.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining: 1.0 at start, 0.0 at end
        cosine = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return final + (initial - final) * cosine
    return func


# ===========================================================================
# SECTION 1 | HYPERPARAMETERS & FILE PATHS
# ===========================================================================

# -- Environment ------------------------------------------------------------
N_ENVS = 4
MAX_EPISODE_STEPS = 4000
SEED = 2024

# -- Coverage grid ----------------------------------------------------------
GRID_N = 10
CELL_SIZE_X = MAP_W / GRID_N
CELL_SIZE_Y = MAP_H / GRID_N
COVERAGE_DIM = GRID_N * GRID_N

# -- Curriculum -------------------------------------------------------------
CURRICULUM_STAGES = [
    (0,         3),
    (800_000,   5),
    (2_000_000, 7),
    (3_400_000, 10),
]

# -- Reward shaping (v5 — rebalanced & anti-spinning) ----------------------
#
# DESIGN PRINCIPLES:
#   1. Per-step rewards live in roughly [-3, +3]. Person-found is the only
#      event that spikes higher, and it's capped at +10 per detection.
#   2. Every undesirable behavior (spinning, idling, revisiting) has an
#      explicit penalty — nothing is "implicitly discouraged."
#   3. Exploration has a smooth gradient, not just a cell-boundary trigger.
#
R_PERSON_FOUND          =  10.0   # ↓ from 1000. Still the dominant positive signal.
R_COLLISION             = -5.0    # ↓ from -500. Episode already terminates; no need
                                  #   to nuke the value function as well.
R_FORMATION_BONUS       =  0.3    # Gentle nudge toward formation keeping.
R_COVERAGE_NEW          =  1.5    # Per newly-covered grid cell.
R_COVERAGE_PROGRESS_EXP =  0.5    # Exponent for progressive bonus: later cells
                                  #   are multiplied by (coverage_frac)^exp, so
                                  #   discovering cell 90/100 is worth ~1.4× cell 10/100.
R_REDISCOVERED          = -0.01   # Mild penalty for rescanning known cells.
R_TIME_PENALTY          = -0.08   # Per-step cost of existing. Makes idle painful.
R_SPIN_PENALTY_SCALE    = -0.5    # Scaled by |omega|/OMEGA_MAX. Full spin = -0.5/step.
R_HEADING_CHANGE_SCALE  = -0.15   # Scaled by |delta_heading|/pi. Discourages jitter.
R_EXPLORATION_GRADIENT  =  0.3    # Reward for reducing distance to nearest uncovered
                                  #   cell (normalized, so max per step ≈ 0.3).
R_NEAR_OBS_EXP_MAX     = -1.5    # Exponential proximity penalty near obstacles.
R_NEAR_WALL_EXP_MAX    = -1.5    # Exponential proximity penalty near walls.
EXP_OBS_ALPHA           =  4.0
EXP_WALL_ALPHA          =  5.0

# -- Safety and control blend -----------------------------------------------
USE_RESIDUAL_CLASSICAL_LEADER = False
RESIDUAL_V_SCALE       = 0.25
RESIDUAL_W_SCALE       = 0.35
SAFETY_SLOW_CLEARANCE  = 1.5
SAFETY_STOP_CLEARANCE  = 0.9
WALL_SAFE_CLEARANCE    = 1.2
COMMAND_SMOOTH_ALPHA   = 0.6
TERMINATE_ON_COLLISION = True

# -- RecurrentPPO hyperparameters -------------------------------------------
LEARNING_RATE  = linear_schedule(3e-4)
N_STEPS        = 2048
# NOTE: For RecurrentPPO, batch_size = number of *sequences* per minibatch.
# With N_ENVS=4 and N_STEPS=2048, total rollout = 8192 steps.
# RecurrentPPO splits these into sequences of length 'n_steps' by default.
# A batch_size of 512 gives 4 minibatches per epoch — good GPU utilisation
# without blowing VRAM on full-sequence backprop.
BATCH_SIZE     = 512
N_EPOCHS       = 10
GAMMA          = 0.995           # ↑ from 0.99. Longer horizon helps the agent
                                 #   value distant person-finds.
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = cosine_entropy_schedule(initial=0.05, final=0.005)
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


# ===========================================================================
# SECTION 2 | GYMNASIUM WRAPPER
# ===========================================================================
class SARGymnasiumWrapper(gym.Env):
    """
    Gymnasium wrapper with hybrid APF+RL control, coverage map, and curriculum.

    Observation layout (length = SAREnvironment.OBS_DIM + COVERAGE_DIM):
      [base env obs..., coverage grid binary flags]
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM = SAREnvironment.OBS_DIM + COVERAGE_DIM

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

        self._step_count: int = 0
        self._prev_found: int = 0
        self._prev_collisions: int = 0
        self._prev_v_l: float = 0.0
        self._prev_w_l: float = 0.0
        self._prev_heading: float = 0.0
        self._prev_dist_to_uncovered: float = 0.0
        self._pre_detected_count: int = 0
        self._active_total: int = N_PERSONS
        self._coverage_grid: np.ndarray = np.zeros(COVERAGE_DIM, dtype=np.float32)

        # Pre-compute cell centers once (used for coverage + exploration gradient)
        x_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_X
        y_centers = (np.arange(GRID_N, dtype=np.float32) + 0.5) * CELL_SIZE_Y
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        self._coverage_cell_centers = np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)

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
        self._active_total = max(1, len(self._env.persons) - self._pre_detected_count)
        self._env.total_found += pre_found

    # --- Observation --------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    def _update_sensor_coverage(self) -> tuple[int, int]:
        centers = self._coverage_cell_centers
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
            angles = np.arctan2(dy, dx)
            rel = self._wrap_to_pi(angles - sensor_hdg)
            in_cone = in_range & (np.abs(rel) <= half_fov)
            observed |= in_cone

        previously_observed = observed & (self._coverage_grid >= 0.5)
        newly_observed = observed & (self._coverage_grid < 0.5)
        new_count = int(np.count_nonzero(newly_observed))
        rediscovered_count = int(np.count_nonzero(previously_observed))
        if new_count > 0:
            self._coverage_grid[newly_observed] = 1.0
        return new_count, rediscovered_count

    def _min_dist_to_uncovered(self, pos: np.ndarray) -> float:
        """Distance from leader to the nearest uncovered grid cell center.

        Returns 0.0 if all cells are covered (nothing to explore).
        """
        uncovered_mask = self._coverage_grid < 0.5
        if not np.any(uncovered_mask):
            return 0.0
        uncovered_centers = self._coverage_cell_centers[uncovered_mask]
        dx = uncovered_centers[:, 0] - float(pos[0])
        dy = uncovered_centers[:, 1] - float(pos[1])
        dists = np.sqrt(dx * dx + dy * dy)
        return float(np.min(dists))

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = raw_obs.copy()

        # Mask undetected person slots (slots start at 15 + 3*N_OBS)
        person_start = 15 + 3 * N_OBS
        for i in range(N_PERSONS):
            base = person_start + 3 * i
            detected = obs[base + 2]
            if detected < 0.5:
                obs[base] = 0.0
                obs[base + 1] = 0.0

        return np.concatenate([obs, self._coverage_grid], dtype=np.float32)

    # --- Reward helpers -----------------------------------------------------

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
        """Exponentially increasing penalty as clearance drops below safe threshold."""
        if clearance >= safe_clearance:
            return 0.0
        deficit = (safe_clearance - clearance) / max(safe_clearance, 1e-6)
        deficit = float(np.clip(deficit, 0.0, 1.5))
        if alpha <= 1e-6:
            scaled = deficit
        else:
            denom = np.expm1(alpha)
            scaled = np.expm1(alpha * deficit) / max(denom, 1e-9)
        scaled = float(np.clip(scaled, 0.0, 1.0))
        return max_penalty * scaled

    # --- Core reward function -----------------------------------------------

    def _shape_reward(
        self,
        info: dict,
        raw_obs: np.ndarray,
        action: np.ndarray,
        new_coverage_cells: int,
        rediscovered_coverage_cells: int,
        min_obs_clearance: float,
        min_wall_clearance: float,
        dist_to_uncovered: float,
    ) -> float:
        reward = 0.0

        # ── 1. PRIMARY OBJECTIVE: find persons ──
        current_found = info["found"]
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            reward += R_PERSON_FOUND * new_detections
        self._prev_found = current_found

        # ── 2. COLLISION (reduced — termination is the main deterrent) ──
        if info["collisions"] > self._prev_collisions:
            reward += R_COLLISION
        self._prev_collisions = info["collisions"]

        # ── 3. FORMATION KEEPING ──
        if info["form_dev_mean"] < 0.5:
            reward += R_FORMATION_BONUS

        # ── 4. COVERAGE — progressive bonus ──
        #   The multiplier grows with coverage_frac so that discovering the
        #   last cells is worth more than the first (harder to reach).
        if new_coverage_cells > 0:
            coverage_frac = float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
            progress_multiplier = 1.0 + coverage_frac ** R_COVERAGE_PROGRESS_EXP
            reward += R_COVERAGE_NEW * float(new_coverage_cells) * progress_multiplier

        if rediscovered_coverage_cells > 0:
            reward += R_REDISCOVERED * float(rediscovered_coverage_cells)

        # ── 5. EXPLORATION GRADIENT (smooth signal toward uncovered cells) ──
        #   Positive reward when the leader moves closer to the nearest
        #   uncovered cell; negative if it moves away.
        if self._prev_dist_to_uncovered > 0 and dist_to_uncovered > 0:
            # Normalise by the map diagonal so the signal is scale-independent
            map_diag = float(np.hypot(MAP_W, MAP_H))
            delta = (self._prev_dist_to_uncovered - dist_to_uncovered) / map_diag
            reward += R_EXPLORATION_GRADIENT * np.clip(delta * 10.0, -1.0, 1.0)

        # ── 6. ANTI-SPINNING: angular velocity penalty ──
        #   |omega| / OMEGA_MAX ∈ [0, 1]. Full spin costs R_SPIN_PENALTY_SCALE.
        omega_fraction = abs(float(action[1]))  # action is already in [-1, 1]
        reward += R_SPIN_PENALTY_SCALE * omega_fraction

        # ── 7. HEADING JITTER penalty ──
        #   Penalise large heading changes between consecutive steps.
        leader = self._env.robots[0]
        heading = float(leader.theta)
        delta_heading = abs(self._wrap_to_pi(
            np.array([heading - self._prev_heading])
        )[0])
        reward += R_HEADING_CHANGE_SCALE * (delta_heading / np.pi)
        self._prev_heading = heading

        # ── 8. PROXIMITY penalties (obstacles & walls) ──
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

        # ── 9. TIME PENALTY (makes idling costly) ──
        reward += R_TIME_PENALTY

        return float(reward)

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
        self._update_sensor_coverage()

        self._step_count = 0
        self._prev_found = self._env.total_found
        self._prev_collisions = 0
        self._prev_v_l = 0.0
        self._prev_w_l = 0.0
        self._prev_heading = float(self._env.robots[0].theta)
        self._prev_dist_to_uncovered = self._min_dist_to_uncovered(
            self._env.robots[0].pos
        )

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(action, -1.0, 1.0)
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()

        # Pure RL leader control
        v_l = float((a[0] + 1.0) / 2.0 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)

        # Safety speed limits near obstacles / walls
        min_obs_clearance = self._min_known_clearance(leader.pos, known_obs)
        min_wall_clearance = self._min_wall_clearance(leader.pos)
        min_clearance = min(min_obs_clearance, min_wall_clearance)
        if min_clearance < SAFETY_STOP_CLEARANCE:
            v_l = min(v_l, 0.15 * V_MAX)
        elif min_clearance < SAFETY_SLOW_CLEARANCE:
            v_l = min(v_l, 0.50 * V_MAX)

        # Smooth leader commands
        v_l = float(
            COMMAND_SMOOTH_ALPHA * self._prev_v_l
            + (1.0 - COMMAND_SMOOTH_ALPHA) * v_l
        )
        w_l = float(
            COMMAND_SMOOTH_ALPHA * self._prev_w_l
            + (1.0 - COMMAND_SMOOTH_ALPHA) * w_l
        )
        v_l = float(np.clip(v_l, 0.0, V_MAX))
        w_l = float(np.clip(w_l, -OMEGA_MAX, OMEGA_MAX))
        self._prev_v_l = v_l
        self._prev_w_l = w_l

        # Follower APF controllers
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)

        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1

        # Coverage update
        new_coverage_cells, rediscovered_coverage_cells = (
            self._update_sensor_coverage()
        )

        # Exploration gradient: distance to nearest uncovered cell
        dist_to_uncovered = self._min_dist_to_uncovered(leader.pos)

        obs = self._build_obs(raw_obs)
        collided_now = info["collisions"] > (self._prev_collisions)

        reward = self._shape_reward(
            info,
            raw_obs,
            a,
            new_coverage_cells,
            rediscovered_coverage_cells,
            min_obs_clearance,
            min_wall_clearance,
            dist_to_uncovered,
        )

        # Update exploration gradient tracker AFTER reward computation
        self._prev_dist_to_uncovered = dist_to_uncovered

        # Termination logic
        active_found = int(np.clip(
            info["found"] - self._pre_detected_count,
            0,
            self._active_total,
        ))
        terminated = active_found >= self._active_total
        if TERMINATE_ON_COLLISION and collided_now:
            terminated = True
        truncated = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

        info["coverage_frac"] = (
            float(np.count_nonzero(self._coverage_grid)) / COVERAGE_DIM
        )
        info["full_found"] = int(info["found"])
        info["full_total"] = int(info["total"])
        info["active_found"] = active_found
        info["active_total"] = self._active_total
        info["pre_detected"] = self._pre_detected_count
        info["found"] = active_found
        info["total"] = self._active_total
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
                    CELL_SIZE_X,
                    CELL_SIZE_Y,
                    linewidth=0,
                    facecolor="#00e676",
                    alpha=0.08,
                    zorder=1,
                )
            )

        self._fig.suptitle(
            "SAR Swarm [RL+APF] "
            f"t={self._env.t:.1f}s "
            f"Active={self._n_active_persons} "
            f"Coverage={int(np.count_nonzero(self._coverage_grid))}/{COVERAGE_DIM}",
            color="#ccc",
            fontsize=9,
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
# SECTION 5 | SAR METRICS CALLBACK
# ===========================================================================
class SARMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", []),
        ):
            if done and "found" in info:
                active_total = max(
                    info.get("active_total", info.get("total", 1)), 1
                )
                active_found = float(np.clip(
                    info.get("active_found", info.get("found", 0)),
                    0,
                    active_total,
                ))
                stage = 1
                for i, (threshold, _) in enumerate(CURRICULUM_STAGES):
                    if self.num_timesteps >= threshold:
                        stage = i + 1

                self.logger.record("sar/persons_found", active_found)
                self.logger.record("sar/persons_total", float(active_total))
                self.logger.record("sar/find_rate", active_found / active_total)
                self.logger.record("sar/collisions", float(info["collisions"]))
                self.logger.record("sar/form_dev_mean", info["form_dev_mean"])
                self.logger.record(
                    "sar/coverage_frac", info.get("coverage_frac", 0.0)
                )
                self.logger.record("sar/curriculum_stage", float(stage))
        return True


# ===========================================================================
# SECTION 6 | TRAINING FUNCTION
# ===========================================================================
def train(device: str = "auto") -> None:
    print("=" * 65)
    print("  SAR SWARM — RecurrentPPO + LSTM (v5 — Reward Overhaul)")
    print("=" * 65)
    print(
        f"  Architecture : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']}) "
        f"+ dense {POLICY_KWARGS['net_arch']}"
    )
    print("  Action space : 2-D leader only (APF followers)")
    print(
        f"  Obs dim      : {SARGymnasiumWrapper.OBS_DIM} "
        f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM})"
    )
    print(f"  Workers      : {N_ENVS}")
    print(f"  Total steps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Device       : {device}")
    print("  Curriculum:")
    for i, (thr, n) in enumerate(CURRICULUM_STAGES):
        print(f"    Stage {i + 1}: {n:>2} persons (from step {thr:>9,})")
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
    train_env.env_method("set_n_active_persons", CURRICULUM_STAGES[0][1])
    print(f"      OK — {N_ENVS} workers ready.\n")

    print("[3/5] Building evaluation environment (10 persons, fixed seed)...")
    eval_env = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)),
        n_envs=1,
    )
    eval_env.env_method("set_n_active_persons", N_PERSONS)
    print("      OK — Eval environment ready (full difficulty).\n")

    print("[4/5] Instantiating RecurrentPPO model...")
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

    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  TensorBoard: tensorboard --logdir ppo_sar_tensorboard\n")
    t0 = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name="RecurrentPPO_SAR",
        )
    except RuntimeError as e:
        print(f"\n[!] Training interrupted by error: {e}")
        print("[!] Attempting emergency model save...")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    model.save(MODEL_PATH)
    print(f"OK — Final model saved to '{MODEL_PATH}.zip'")
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

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        cumulative_rew = 0.0
        done = False
        step_num = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)
            cumulative_rew += reward
            step_num += 1
            if step_num % 3 == 0:
                env.render()
            done = terminated or truncated

        end_reason = "SUCCESS" if info["found"] >= info["total"] else "TIMEOUT"
        if info["collisions"] > 0 and not (info["found"] >= info["total"]):
            end_reason = "COLLISION"

        results.append(
            {
                "episode": ep + 1,
                "reward": cumulative_rew,
                "persons_found": info["found"],
                "persons_total": info["total"],
                "collisions": info["collisions"],
                "form_dev_mean": info["form_dev_mean"],
                "coverage_frac": info.get("coverage_frac", 0.0),
                "steps": step_num,
                "end_reason": end_reason,
            }
        )

        print(f"  End condition   : {end_reason}")
        print(f"  Steps           : {step_num} / {MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {info['found']} / {info['total']}")
        print(f"  Collisions      : {info['collisions']}")
        print(f"  Form. dev (avg) : {info['form_dev_mean']:.3f} m")
        print(f"  Map coverage    : {info.get('coverage_frac', 0.0) * 100:.1f}%")
        if wait_for_input:
            input("  [Press Enter for next episode] ")
        env.close()

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
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run evaluation without waiting for Enter between episodes",
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
        train(device=args.device)

    elif args.mode == "eval":
        evaluate(
            n_episodes=args.episodes,
            device=args.device,
            wait_for_input=not args.no_prompt,
        )

    elif args.mode == "both":
        train(device=args.device)
        evaluate(
            n_episodes=args.episodes,
            device=args.device,
            wait_for_input=not args.no_prompt,
        )
