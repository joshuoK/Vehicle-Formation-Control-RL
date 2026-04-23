"""
train_rl.py  (v3 — LSTM + Curriculum + Coverage Map + Partial Observability Fix)
=================================================================================
Four architectural improvements that make 100% person discovery achievable:

1.  VISITED-CELL COVERAGE MAP
    A 10x10 grid (each cell = 2x2 m) is maintained in the wrapper and appended
    to every observation.  Cells flip from 0 to 1 the first time the leader
    enters them, and each new cell visited earns +R_COVERAGE reward.  The agent
    can now "see" where it has and has not been, so it learns systematic sweeping
    instead of random wandering.

2.  PARTIAL OBSERVABILITY FIX
    The raw environment always writes person (x,y) into the obs vector even
    before detection.  The wrapper now zeros out the position of any undetected
    person so the agent must earn that information by actually visiting the area.
    The detected-flag component (0.0 / 1.0) is kept so the agent knows how many
    targets remain.

3.  RECURRENT POLICY (LSTM)
    Standard MLP policies have no memory across time-steps.  Switching to
    RecurrentPPO (sb3_contrib) with an LSTM cell gives the agent persistent
    hidden state across the whole episode.

    Install: pip install sb3-contrib

4.  CURRICULUM LEARNING
    Stages ramp difficulty gradually so the agent gets positive signal early:

        Stage 1 (  0 -> 500k steps):  3 active persons
        Stage 2 (500k -> 1.2M steps): 5 active persons
        Stage 3 (1.2M -> 2.5M steps): 7 active persons
        Stage 4 (2.5M -> 5M steps):  10 active persons

    The eval environment always runs at full difficulty (10 persons).

All earlier fixes are retained: hybrid APF followers, seed diversity,
no crash-on-collision, improved reward shaping.

Quick-start
-----------
    pip install stable-baselines3[extra] sb3-contrib gymnasium matplotlib numpy tqdm

    python train_rl.py --mode train
    python train_rl.py --mode eval
    python train_rl.py --mode both
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
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util     import make_vec_env
from stable_baselines3.common.vec_env      import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks    import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor      import Monitor
from stable_baselines3.common.env_checker  import check_env

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError(
        "sb3-contrib is required for the LSTM policy.\n"
        "Install with:  pip install sb3-contrib"
    )

sys.path.insert(0, os.path.dirname(__file__))
from sar_environment import (
    SAREnvironment,
    V_MAX, OMEGA_MAX,
    N_PERSONS, N_OBS,
    FORM_OFFSET,
    MAP_W, MAP_H,
)
from sar_classical_controller import APFFollowerCtrl


# ---------------------------------------------------------------------------
# Cross-platform multiprocessing
# ---------------------------------------------------------------------------
multiprocessing.freeze_support()
_IS_WINDOWS      = (platform.system() == 'Windows')
_MP_START_METHOD = 'spawn' if _IS_WINDOWS else 'fork'


# ===========================================================================
# SECTION 1  |  HYPERPARAMETERS & FILE PATHS
# ===========================================================================

# -- Environment -------------------------------------------------------------
N_ENVS            = 4
MAX_EPISODE_STEPS = 4000     # ~200 s at DT=0.05
SEED              = 2024

# -- Coverage grid -----------------------------------------------------------
GRID_N       = 10                    # 10x10 = 100 cells
CELL_SIZE    = MAP_W / GRID_N        # 2.0 m per cell
COVERAGE_DIM = GRID_N * GRID_N      # 100 extra obs features

# -- Curriculum  [(start_timestep, n_active_persons)] -----------------------
CURRICULUM_STAGES = [
    (        0,  3),
    (  500_000,  5),
    (1_200_000,  7),
    (2_500_000, 10),
]

# -- Reward shaping ----------------------------------------------------------
R_PERSON_FOUND    =  50.0   # per newly detected rescue target
R_COLLISION       = -40.0   # per new collision (no episode termination)
R_FORWARD         =   0.2   # per step leader moves forward
R_FORMATION_BONUS =   1.0   # when form_dev < 0.5 m
R_COVERAGE        =   0.5   # first visit to a new 2x2 m grid cell
R_TIME_PENALTY    =  -0.02  # per step

# -- LSTM / RecurrentPPO hyperparameters ------------------------------------
LEARNING_RATE  = 3e-4
N_STEPS        = 512    # rollout per worker; shorter suits LSTM
BATCH_SIZE     = 128    # must divide N_ENVS * N_STEPS = 2048
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
    net_arch           = dict(pi=[128], vf=[128]),
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
    Gymnasium wrapper with hybrid APF+RL control, coverage map, and curriculum.

    Observation layout  (float32, length = SAREnvironment.OBS_DIM + COVERAGE_DIM)
    ------------------------------------------------------------------------------
      [0:15]    Robot poses/velocities  (5 values x 3 robots)
      [15:36]   Known obstacles         (3 values x N_OBS slots, zero-padded)
      [36:66]   Person slots            (3 values x N_PERSONS)
                  [px/W, py/H, detected]
                  px/W and py/H are ZEROED for undetected persons.
      [66:166]  Coverage grid           (100 binary values: 0=unvisited, 1=visited)

    Action space  (2-D, normalised [-1, 1])
      [0]  v_leader  -> [0, V_MAX]
      [1]  w_leader  -> [-OMEGA_MAX, OMEGA_MAX]

    Followers are driven by classical APF every step.
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM  = SAREnvironment.OBS_DIM + COVERAGE_DIM   # 66 + 100 = 166

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed        = seed
        self._episode_count    = 0
        self._n_active_persons = N_PERSONS

        self._env    = SAREnvironment(seed=seed)
        self._fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
        self._fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._step_count:      int       = 0
        self._prev_found:      int       = 0
        self._prev_collisions: int       = 0
        self._coverage_grid:   np.ndarray = np.zeros(COVERAGE_DIM, dtype=np.float32)
        self._visited_cells:   set       = set()

    # ── Curriculum ────────────────────────────────────────────────────────────

    def set_n_active_persons(self, n: int) -> None:
        """Update active persons count. Called by CurriculumCallback."""
        self._n_active_persons = int(np.clip(n, 1, N_PERSONS))

    def _apply_curriculum(self) -> None:
        """Pre-detect excess persons so only _n_active_persons are real targets."""
        pre_found = 0
        for i, p in enumerate(self._env.persons):
            if i >= self._n_active_persons and not p.detected:
                p.detected = True
                pre_found += 1
        self._env.total_found += pre_found

    # ── Observation ───────────────────────────────────────────────────────────

    def _cell_index(self, x: float, y: float) -> int:
        ci = int(np.clip(x / CELL_SIZE, 0, GRID_N - 1))
        cj = int(np.clip(y / CELL_SIZE, 0, GRID_N - 1))
        return ci * GRID_N + cj

    def _build_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Apply partial-observability mask then append coverage grid.
        Undetected person positions are zeroed so the agent cannot navigate
        to them without first exploring the area.
        """
        obs = raw_obs.copy()

        # Person slots start at index 15 + 3*N_OBS
        person_start = 15 + 3 * N_OBS
        for i in range(N_PERSONS):
            base     = person_start + 3 * i
            detected = obs[base + 2]
            if detected < 0.5:
                obs[base]     = 0.0
                obs[base + 1] = 0.0

        return np.concatenate([obs, self._coverage_grid], dtype=np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _shape_reward(self, info: dict, raw_obs: np.ndarray,
                      new_cell: bool) -> float:
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

        # 3. Forward motion bonus (raw_obs[3] = v_leader / V_MAX)
        if float(raw_obs[3]) > 0.05:
            reward += R_FORWARD

        # 4. Formation quality bonus
        if info['form_dev_mean'] < 0.5:
            reward += R_FORMATION_BONUS

        # 5. Coverage bonus — first visit to a new grid cell
        if new_cell:
            reward += R_COVERAGE

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

        # Reset coverage grid
        self._coverage_grid = np.zeros(COVERAGE_DIM, dtype=np.float32)
        self._visited_cells = set()
        leader = self._env.robots[0]
        idx    = self._cell_index(leader.x, leader.y)
        self._visited_cells.add(idx)
        self._coverage_grid[idx] = 1.0

        # prev_found must start at pre-detected count so reward delta is correct
        self._step_count      = 0
        self._prev_found      = self._env.total_found
        self._prev_collisions = 0

        raw_obs = self._env._build_obs()
        return self._build_obs(raw_obs), {}

    def step(self, action: np.ndarray) -> tuple:
        # Rescale leader action to physical limits
        a   = np.clip(action, -1.0, 1.0)
        v_l = float((a[0] + 1.0) / 2.0 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)

        # APF follower commands
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)

        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1

        # Update coverage grid
        new_leader = self._env.robots[0]
        cell_idx   = self._cell_index(new_leader.x, new_leader.y)
        new_cell   = cell_idx not in self._visited_cells
        if new_cell:
            self._visited_cells.add(cell_idx)
            self._coverage_grid[cell_idx] = 1.0

        obs    = self._build_obs(raw_obs)
        reward = self._shape_reward(info, raw_obs, new_cell)

        terminated = int(info['found']) >= int(info['total'])
        truncated  = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)

        info['coverage_frac'] = float(len(self._visited_cells)) / COVERAGE_DIM

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

        # Overlay visited cells
        for idx in self._visited_cells:
            ci = idx // GRID_N
            cj = idx  % GRID_N
            self._ax.add_patch(plt.Rectangle(
                (ci * CELL_SIZE, cj * CELL_SIZE), CELL_SIZE, CELL_SIZE,
                linewidth=0, facecolor='#00e676', alpha=0.08, zorder=1,
            ))

        self._fig.suptitle(
            f'SAR Swarm [RL+APF]  t={self._env.t:.1f}s  '
            f'Active={self._n_active_persons}  '
            f'Coverage={len(self._visited_cells)}/{COVERAGE_DIM}',
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

    At each transition it updates every training worker via env_method().
    The eval environment is NOT updated — it always runs at full difficulty
    so TensorBoard curves reflect true mission performance.
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

def train() -> None:
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
    print("  SAR SWARM  —  RecurrentPPO + LSTM  (v3)")
    print("=" * 65)
    print(f"  Architecture    : LSTM(hidden={POLICY_KWARGS['lstm_hidden_size']})"
          f" + dense {POLICY_KWARGS['net_arch']}")
    print(f"  Action space    : 2-D leader only (APF followers)")
    print(f"  Obs dim         : {SARGymnasiumWrapper.OBS_DIM} "
          f"(env {SAREnvironment.OBS_DIM} + coverage {COVERAGE_DIM})")
    print(f"  Workers         : {N_ENVS}")
    print(f"  Total steps     : {TOTAL_TIMESTEPS:,}")
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

def evaluate(n_episodes: int = N_EVAL_EPISODES) -> None:
    """
    Load the saved model and run n_episodes visualised test episodes.

    LSTM state is carried between steps and reset at episode boundaries
    by passing episode_starts to model.predict().
    """
    model_file = f"{MODEL_PATH}.zip"
    if not os.path.exists(model_file):
        print(f"❌  Model '{model_file}' not found.")
        print("   Run  python train_rl.py --mode train  first.")
        return

    print(f"\nLoading model from '{model_file}'...")
    model = RecurrentPPO.load(model_file)
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
    print(f"  Mean reward      : {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
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
        train()

    elif args.mode == "eval":
        evaluate(n_episodes=args.episodes)

    elif args.mode == "both":
        train()
        evaluate(n_episodes=args.episodes)
