"""
train_rl_v0.py
===========
Deep Reinforcement Learning training script for the SAR Swarm environment.
Uses Proximal Policy Optimisation (PPO) from Stable-Baselines3 (SB3).

Quick-start
-----------
    # Install dependencies (once)
    pip install stable-baselines3[extra] gymnasium matplotlib numpy

    # Train (writes model + TensorBoard logs)
    python train_rl.py --mode train

    # Evaluate a saved model (renders 3 test episodes)
    python train_rl.py --mode eval

    # Train then auto-evaluate
    python train_rl.py --mode both

Design rationale
----------------
SAREnvironment uses a *legacy* 4-tuple step() interface:
    (obs, reward, done, info)

Modern Gymnasium expects a 5-tuple:
    (obs, reward, terminated, truncated, info)

The SARGymnasiumWrapper bridges that gap cleanly, adds custom reward
shaping, enforces a step-count time limit, and exposes proper
observation_space / action_space attributes so SB3 can validate the env.

Reward shaping philosophy
-------------------------
The wrapper REPLACES the base environment reward entirely.  The design
priorities from highest to lowest are:

  1. Find people  (+50 / person)   -- the primary mission objective.
                                      Large sparse signal ensures the agent
                                      never ignores the main goal.

  2. Stay safe    (-500 + done)    -- immediate episode termination on
                                      collision.  The -500 overwhelms any
                                      accumulated +50 bonuses so the safest
                                      policy is always to avoid obstacles.

  3. Keep moving  (+0.1 / step)   -- tiny dense exploration bonus.  Without
                                      this, a zero-velocity policy scores 0
                                      per step and the agent may learn to do
                                      nothing.

  4. Hold formation (-0.5 * dev)   -- dense penalty for V-shape deviation.
                                      Encourages the agent to keep followers
                                      in their slots; scales with severity.

  5. Path freedom  (no penalty)   -- path_dev is deliberately ignored.
                                      The RL agent should discover its own
                                      search trajectory; imposing the sweep
                                      path would handicap the policy vs the
                                      classical baseline.

Parallel training
-----------------
Training uses SubprocVecEnv to run N_ENVS copies of the environment in
separate processes.  This multiplies the effective sampling throughput
by N_ENVS with no extra wall-clock cost per step.

File outputs
------------
  ppo_swarm_agent.zip          -- final trained model (loadable with PPO.load)
  ppo_sar_tensorboard/         -- TensorBoard event files; run:
                                    tensorboard --logdir ppo_sar_tensorboard
  checkpoints/                 -- intermediate saves every CHECKPOINT_FREQ steps
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
# Set backend before pyplot is imported.
# On Windows there is no DISPLAY env var; 'TkAgg' is the standard
# interactive backend.  Fall back to 'Agg' (file-only) if Tk is absent.
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# Stable-Baselines3 core
from stable_baselines3 import PPO
from stable_baselines3.common.env_util     import make_vec_env
from stable_baselines3.common.vec_env      import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks    import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor      import Monitor
from stable_baselines3.common.env_checker  import check_env

# Our environment -- must be in the same directory (or on PYTHONPATH)
sys.path.insert(0, os.path.dirname(__file__))
from sar_environment import SAREnvironment, V_MAX, OMEGA_MAX, N_PERSONS


# ---------------------------------------------------------------------------
# Cross-platform multiprocessing setup
# ---------------------------------------------------------------------------
# Windows only supports the 'spawn' start method.  Calling freeze_support()
# is required when the script is frozen by PyInstaller; it is a harmless
# no-op in normal Python execution, so we call it unconditionally here at
# module level (before any worker processes are created).
#
# 'fork' is unavailable on Windows -- passing it to SubprocVecEnv raises:
#   ValueError: cannot find context for 'fork'
# We detect the OS once and reference _MP_START_METHOD throughout.
#
# IMPORTANT: on Windows, 'spawn' re-imports this entire script in every
# worker process.  All process-creation code MUST live inside the
# `if __name__ == '__main__':` guard at the bottom of this file so workers
# don't recursively try to spawn more workers.  That guard is already in
# place -- this comment is here to explain *why* it is mandatory on Windows.
multiprocessing.freeze_support()
_IS_WINDOWS      = (platform.system() == 'Windows')
_MP_START_METHOD = 'spawn' if _IS_WINDOWS else 'fork'


# ===========================================================================
# SECTION 1  |  HYPERPARAMETERS & FILE PATHS
# All tunable values are constants here; nothing is buried in the code below.
# ===========================================================================

# -- Environment -------------------------------------------------------------
N_ENVS           = 4        # parallel worker processes for training
MAX_EPISODE_STEPS = 4000    # ~200 s of sim time at DT=0.05; then truncated=True
SEED             = 2024     # base seed; workers get seed+rank for diversity

# -- Reward shaping ----------------------------------------------------------
R_PERSON_FOUND   =  50.0   # reward per newly detected rescue target
R_COLLISION      = -500.0  # applied once when a new collision is detected
R_FORMATION      = -0.5    # multiplied by form_dev_mean each step
R_FORWARD        =  0.1    # flat bonus for any non-zero forward movement

# -- PPO hyperparameters -----------------------------------------------------
# These are reasonable defaults for a continuous-action, medium-complexity env.
# Tune them if training plateaus or is unstable.
LEARNING_RATE    = 3e-4    # Adam LR; reduce if training is noisy
N_STEPS          = 2048    # rollout length per worker per update
BATCH_SIZE       = 256     # mini-batch size for gradient update
N_EPOCHS         = 10      # passes over each rollout buffer
GAMMA            = 0.99    # discount factor (long-horizon tasks need high γ)
GAE_LAMBDA       = 0.95    # GAE-λ smoothing for advantage estimates
CLIP_RANGE       = 0.2     # PPO clipping parameter (standard value)
ENT_COEF         = 0.005   # entropy bonus; keeps policy from collapsing early
VF_COEF          = 0.5     # value-function loss weight
MAX_GRAD_NORM    = 0.5     # gradient clipping for stability

# -- Policy network architecture ---------------------------------------------
# Two hidden layers of 256 each.  The SAR obs vector has 66 features, so 256
# is wide enough to represent the relevant correlations without overfitting.
POLICY_KWARGS = dict(
    net_arch = dict(pi=[256, 256], vf=[256, 256])
)

# -- Training schedule -------------------------------------------------------
TOTAL_TIMESTEPS    = 1_000_000   # total environment steps across all workers
CHECKPOINT_FREQ    = 50_000      # save a checkpoint every N steps (single env)
EVAL_FREQ          = 20_000      # run eval callback every N steps (single env)
EVAL_EPISODES      = 5           # episodes per eval callback run
N_EVAL_EPISODES    = 3           # episodes in the final manual eval loop

# -- File paths --------------------------------------------------------------
MODEL_PATH         = "ppo_swarm_agent"          # .zip appended automatically
TENSORBOARD_LOG    = "./ppo_sar_tensorboard/"
CHECKPOINT_DIR     = "./checkpoints/"


# ===========================================================================
# SECTION 2  |  GYMNASIUM WRAPPER
# Bridges the legacy 4-tuple SAREnvironment to modern Gymnasium API,
# injects custom reward shaping, and enforces the episode step limit.
# ===========================================================================

class SARGymnasiumWrapper(gym.Env):
    """
    Gymnasium-compliant wrapper around SAREnvironment.

    What this class adds on top of the raw environment
    --------------------------------------------------
    1. observation_space / action_space  -- required by SB3 for validation.
    2. 5-tuple step() return             -- (obs, reward, terminated, truncated, info).
    3. 2-tuple reset() return            -- (obs, info), as Gymnasium demands.
    4. Custom reward shaping             -- see module docstring for rationale.
    5. Episode time limit                -- truncated=True after MAX_EPISODE_STEPS.
    6. Collision termination             -- terminated=True + large penalty.

    Observation space
    -----------------
    The raw environment outputs a float32 vector of length OBS_DIM where every
    element is already normalised to approximately [-1, 1].  We declare it as a
    Box space with those bounds so SB3's normalisation layers don't interfere.

    Action space
    ------------
    Six continuous actions: [v_l, w_l, v_f1, w_f1, v_f2, w_f2].
    We declare the Box with bounds [-1, 1] and rescale inside step() so the
    policy network works with a symmetrical output range regardless of the
    physical velocity limits.  This is standard practice for continuous control.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        # The underlying environment that holds all simulation state
        self._env = SAREnvironment(seed=seed)

        # ── Observation space ──────────────────────────────────────────────
        # OBS_DIM is a class attribute on SAREnvironment, computed from
        # N_OBS and N_PERSONS constants.  Float32, all values in [-1, 1].
        obs_dim = SAREnvironment.OBS_DIM
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (obs_dim,),
            dtype = np.float32,
        )

        # ── Action space ───────────────────────────────────────────────────
        # Six actions normalised to [-1, 1].  Rescaling to physical limits
        # happens in _rescale_actions() so the policy always sees [-1, 1].
        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (6,),
            dtype = np.float32,
        )

        # ── Episode bookkeeping ────────────────────────────────────────────
        self._step_count:    int = 0     # steps taken in current episode
        self._prev_found:    int = 0     # persons found at last step (for delta)
        self._prev_collisions: int = 0   # collisions at last step (for delta)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rescale_actions(self, action: np.ndarray) -> list:
        """
        Rescale normalised [-1, 1] policy outputs to physical velocity limits.

        Layout of the 6-element action vector:
          [0] v_leader   : forward speed    -> scaled to [0, V_MAX]
          [1] w_leader   : angular rate     -> scaled to [-OMEGA_MAX, OMEGA_MAX]
          [2] v_f1       : follower-1 fwd   -> [0, V_MAX]
          [3] w_f1       : follower-1 ang   -> [-OMEGA_MAX, OMEGA_MAX]
          [4] v_f2       : follower-2 fwd   -> [0, V_MAX]
          [5] w_f2       : follower-2 ang   -> [-OMEGA_MAX, OMEGA_MAX]

        Forward velocities are clipped to [0, V_MAX] (no reversing) because
        the robots are designed for forward exploration; allowing reverse speeds
        dramatically expands the policy search space without benefit.

        Angular velocities span the full symmetric range [-OMEGA_MAX, OMEGA_MAX].
        """
        a = np.clip(action, -1.0, 1.0)   # defensive clip before rescaling

        # Even indices = linear velocity:  map [-1,1] -> [0, V_MAX]
        # Odd  indices = angular velocity: map [-1,1] -> [-OMEGA_MAX, OMEGA_MAX]
        v_l  = float((a[0] + 1.0) / 2.0 * V_MAX)
        w_l  = float(a[1] * OMEGA_MAX)
        v_f1 = float((a[2] + 1.0) / 2.0 * V_MAX)
        w_f1 = float(a[3] * OMEGA_MAX)
        v_f2 = float((a[4] + 1.0) / 2.0 * V_MAX)
        w_f2 = float(a[5] * OMEGA_MAX)
        return [v_l, w_l, v_f1, w_f1, v_f2, w_f2]

    def _shape_reward(self, info: dict, raw_obs: np.ndarray) -> float:
        """
        Compute the shaped step reward from the info dict.

        The base environment reward is DISCARDED — we compute everything
        from the info dict so the shaping is transparent and easy to change.

        Reward components (ordered by priority)
        ----------------------------------------
        1. Person found  (+R_PERSON_FOUND per new detection)
           Calculated as the DELTA in info['found'] since the last step so
           we never double-count detections across steps.

        2. Collision      (R_COLLISION, applied once per new event)
           Similarly computed as a delta so a lingering collision does not
           spam the penalty every step.  The episode also terminates.

        3. Forward motion (+R_FORWARD if leader is making headway)
           Uses the leader's normalised linear velocity from the observation
           vector (index 3 = v_leader / V_MAX).  A small positive threshold
           (> 0.05) ignores near-zero movements.

        4. Formation      (R_FORMATION * form_dev_mean per step)
           Dense per-step penalty proportional to how far the followers have
           drifted from their ideal V-slots.
        """
        reward = 0.0

        # ── 1. Person detection delta ──────────────────────────────────────
        current_found  = info['found']
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            reward += R_PERSON_FOUND * new_detections
        self._prev_found = current_found

        # ── 2. Collision delta ─────────────────────────────────────────────
        current_collisions = info['collisions']
        new_collision      = current_collisions > self._prev_collisions
        if new_collision:
            reward += R_COLLISION
        self._prev_collisions = current_collisions

        # ── 3. Forward exploration bonus ───────────────────────────────────
        # obs index 3 = v_leader / V_MAX (normalised); threshold at 0.05
        leader_v_norm = float(raw_obs[3])
        if leader_v_norm > 0.05:
            reward += R_FORWARD

        # ── 4. Formation quality penalty ───────────────────────────────────
        reward += R_FORMATION * info['form_dev_mean']

        return float(reward)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Start a new episode.

        Gymnasium reset() must return (obs, info).  The base env returns
        only obs, so we add an empty info dict.  Episode counters and
        delta trackers are all reset here so reward shaping starts clean.

        Parameters
        ----------
        seed    : If provided, override the env seed for this episode.
                  Enables reproducible test runs from the eval loop.
        options : Unused; accepted for API compatibility.
        """
        if seed is not None:
            self._env.seed = seed   # forward seed to the underlying env

        obs                   = self._env.reset()  # returns float32 ndarray
        self._step_count      = 0
        self._prev_found      = 0
        self._prev_collisions = 0

        # Return (obs, info) as Gymnasium requires
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the simulation by one DT tick with the given action.

        Returns the standard Gymnasium 5-tuple:
          obs        : float32 observation vector
          reward     : shaped scalar reward (see _shape_reward)
          terminated : True if episode ended due to collision or mission complete
          truncated  : True if episode ended due to MAX_EPISODE_STEPS
          info       : full metrics dict from SAREnvironment.get_metrics()

        Termination conditions
        ----------------------
        terminated = True when:
          (a) a new collision occurs -> immediate safety termination, or
          (b) all persons are found  -> mission success

        truncated = True when:
          (c) step count reaches MAX_EPISODE_STEPS -> time limit

        Both terminated and truncated can only be True one at a time because
        the conditions are checked in priority order: collision first, then
        success, then time limit.
        """
        # Rescale normalised policy output to physical velocity commands
        physical_actions = self._rescale_actions(action)

        # Step the base environment (legacy 4-tuple interface)
        raw_obs, _base_reward, _done, info = self._env.step(physical_actions)

        self._step_count += 1

        # ── Compute shaped reward ──────────────────────────────────────────
        reward = self._shape_reward(info, raw_obs)

        # ── Determine termination / truncation ────────────────────────────
        # New collision: immediate episode end with safety penalty already
        # applied inside _shape_reward above.
        new_collision = info['collisions'] > (self._prev_collisions - 1)
        # Note: _prev_collisions was already updated in _shape_reward so we
        # compare against the UPDATED value minus the increment.
        # Simpler and more robust: re-derive the delta here.
        collision_this_step = (info['collisions'] >
                               (self._prev_collisions -
                                (1 if info['collisions'] > self._prev_collisions - 1 else 0)))

        # Clearest derivation: terminated if any new collision happened
        # (_shape_reward already updated self._prev_collisions)
        terminated = False

        # Check for new collision by inspecting the reward; collision caused
        # R_COLLISION to be added, so cross-check directly:
        if info['collisions'] > 0 and reward <= R_COLLISION + 0.001:
            # A collision penalty was applied this step => terminate
            terminated = True

        # Mission success (all persons found)
        if info['found'] >= info['total']:
            terminated = True

        # Time-limit truncation (only if not already terminated)
        truncated = False
        if not terminated and self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        return raw_obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Delegate to the base environment's matplotlib renderer.

        The base render() requires an Axes object and formation-target
        vectors.  We supply a persistent figure/axes pair and pass None
        for the optional formation targets (they are omitted in eval mode
        where the controller targets are not available).
        """
        # Lazy-create figure so render() can be called any number of times
        if not hasattr(self, '_fig'):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor('#12121f')
            plt.ion()

        self._env.render(self._ax, f1_tgt=None, f2_tgt=None, wp_idx=0)
        self._fig.suptitle(
            f'SAR Swarm  [RL Agent]  ·  t = {self._env.t:.1f} s',
            color='#ccc', fontsize=10,
        )
        plt.pause(0.001)

    def close(self) -> None:
        """Clean up matplotlib figure if it was opened by render()."""
        if hasattr(self, '_fig'):
            plt.close(self._fig)


# ===========================================================================
# SECTION 3  |  ENVIRONMENT FACTORY
# SB3's make_vec_env needs a callable that returns a new env instance.
# We use a closure to vary the seed per worker so each process explores a
# different part of the state space.
# ===========================================================================

def make_env(rank: int, base_seed: int = SEED) -> callable:
    """
    Return a *thunk* (zero-argument callable) that constructs one wrapped env.

    make_vec_env / SubprocVecEnv call this thunk in each worker process.
    Using rank + base_seed as the per-worker seed ensures diverse obstacle
    layouts across workers, which prevents the policy over-fitting to a single
    map configuration during training.

    Parameters
    ----------
    rank      : worker index (0, 1, ..., N_ENVS-1)
    base_seed : base random seed; actual seed = base_seed + rank
    """
    def _init() -> gym.Env:
        env = SARGymnasiumWrapper(seed=base_seed + rank)
        # Monitor wraps the env to log episode rewards/lengths to CSV files,
        # which SB3 uses for TensorBoard 'rollout/ep_rew_mean' curves.
        env = Monitor(env)
        return env
    return _init


# ===========================================================================
# SECTION 4  |  CUSTOM TENSORBOARD CALLBACK
# Logs extra per-episode metrics that are not tracked by SB3 by default,
# such as persons_found, collisions, and formation_dev_mean.
# ===========================================================================

class SARMetricsCallback(BaseCallback):
    """
    Log SAR-specific performance metrics to TensorBoard every episode.

    SB3's built-in logging only captures episode reward and length.
    This callback hooks into the step loop and records the custom info dict
    values so you can plot mission-quality curves alongside the reward curve.

    Metrics logged (prefix: sar/)
    ------------------------------
    sar/persons_found       -- fraction of persons found [0, 1]
    sar/collisions          -- total collision count for the episode
    sar/form_dev_mean       -- mean formation deviation [m]
    sar/path_dev_mean       -- mean path deviation [m]  (for post-hoc analysis)
    sar/obs_discovered      -- fraction of obstacles discovered
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # Running accumulators reset at the start of each episode
        self._ep_persons_found:  list[float] = []
        self._ep_collisions:     list[float] = []
        self._ep_form_dev:       list[float] = []
        self._ep_path_dev:       list[float] = []
        self._ep_obs_discovered: list[float] = []

    def _on_step(self) -> bool:
        """
        Called by SB3 after every environment step across all workers.

        self.locals['infos'] is a list of info dicts, one per parallel env.
        self.locals['dones'] flags which envs just ended their episode.

        We log SAR metrics only at episode boundaries (when done=True) to
        keep TensorBoard plots as 'per episode' rather than 'per step'.
        """
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for info, done in zip(infos, dones):
            if done and 'found' in info:
                total = max(info.get('total', 1), 1)  # guard /0
                obs_t = max(info.get('obs_total', 1), 1)

                self.logger.record('sar/persons_found',
                                   info['found'] / total)
                self.logger.record('sar/collisions',
                                   float(info['collisions']))
                self.logger.record('sar/form_dev_mean',
                                   info['form_dev_mean'])
                self.logger.record('sar/path_dev_mean',
                                   info['path_dev_mean'])
                self.logger.record('sar/obs_discovered',
                                   info['obs_discovered'] / obs_t)

        return True   # returning False would abort training


# ===========================================================================
# SECTION 5  |  TRAINING FUNCTION
# ===========================================================================

def train() -> None:
    """
    Set up vectorised environments, build the PPO model, attach callbacks,
    run training, and save the final model.

    Training outline
    ----------------
    1. Sanity-check the wrapper with SB3's check_env() utility.
    2. Build SubprocVecEnv with N_ENVS parallel workers, each seeded
       differently so the policy sees diverse map configurations.
    3. Build a separate single-env VecEnv for the EvalCallback so
       evaluation always runs on a fixed, reproducible map (seed=SEED+99).
    4. Instantiate PPO with MlpPolicy and the hyperparameters above.
    5. Attach three callbacks:
         - SARMetricsCallback  : logs sar/* metrics to TensorBoard
         - EvalCallback        : periodic evaluation + best-model save
         - CheckpointCallback  : intermediate saves every CHECKPOINT_FREQ steps
    6. Call model.learn() for TOTAL_TIMESTEPS steps.
    7. Save final model to MODEL_PATH.zip.

    TensorBoard
    -----------
    Launch with:  tensorboard --logdir ppo_sar_tensorboard
    Key curves to monitor:
      rollout/ep_rew_mean  -- episode reward (should trend upward)
      sar/persons_found    -- fraction of people rescued (0 -> 1 goal)
      sar/collisions       -- should trend toward 0
      sar/form_dev_mean    -- should decrease as agent learns formation
    """
    print("=" * 60)
    print("  SAR SWARM  —  PPO TRAINING")
    print("=" * 60)
    print(f"  Workers         : {N_ENVS}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Max ep steps    : {MAX_EPISODE_STEPS}")
    print(f"  Model output    : {MODEL_PATH}.zip")
    print(f"  TensorBoard dir : {TENSORBOARD_LOG}")
    print("=" * 60)

    # ── Step 1: Sanity-check one instance of the wrapper ──────────────────
    # check_env() runs a series of assertions that catch common mistakes
    # (wrong dtype, observation out of bounds, incorrect reset return, etc.)
    print("\n[1/5] Checking environment wrapper with SB3 check_env()...")
    _check_env = SARGymnasiumWrapper(seed=SEED)
    check_env(_check_env, warn=True)
    _check_env.close()
    print("      ✅  Wrapper is compliant with Gymnasium API.\n")

    # ── Step 2: Build vectorised training environments ────────────────────
    print("[2/5] Spawning training environments...")
    train_env = SubprocVecEnv(
        [make_env(rank=i, base_seed=SEED) for i in range(N_ENVS)],
        start_method=_MP_START_METHOD,   # 'fork' on Linux/macOS, 'spawn' on Windows
    )
    # VecMonitor aggregates episode stats from all workers into one stream
    train_env = VecMonitor(train_env)
    print(f"      ✅  {N_ENVS} workers ready.\n")

    # ── Step 3: Build evaluation environment ──────────────────────────────
    # Fixed seed (SEED+99) ensures eval episodes are identical across runs,
    # giving a stable signal for the EvalCallback reward curve.
    print("[3/5] Building evaluation environment (fixed seed)...")
    eval_env = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99)),
        n_envs=1,
    )
    print("      ✅  Eval environment ready.\n")

    # ── Step 4: Build PPO model ────────────────────────────────────────────
    print("[4/5] Instantiating PPO model...")
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = LEARNING_RATE,
        n_steps         = N_STEPS,          # steps per worker before update
        batch_size      = BATCH_SIZE,        # SGD mini-batch size
        n_epochs        = N_EPOCHS,          # passes over the rollout buffer
        gamma           = GAMMA,             # discount factor
        gae_lambda      = GAE_LAMBDA,        # GAE smoothing coefficient
        clip_range      = CLIP_RANGE,        # PPO epsilon clipping
        ent_coef        = ENT_COEF,          # exploration entropy bonus
        vf_coef         = VF_COEF,           # value-function loss weight
        max_grad_norm   = MAX_GRAD_NORM,     # gradient clipping threshold
        policy_kwargs   = POLICY_KWARGS,     # network architecture
        tensorboard_log = TENSORBOARD_LOG,   # TensorBoard log directory
        verbose         = 1,                 # print training progress
        seed            = SEED,
    )

    # Print model summary for confirmation
    total_params = sum(
        p.numel() for p in model.policy.parameters()
    )
    print(f"      ✅  PPO model built. "
          f"Policy parameters: {total_params:,}\n")

    # ── Step 5: Assemble callbacks ────────────────────────────────────────
    print("[5/5] Configuring callbacks...")

    # a) Custom SAR metrics logger
    sar_metrics_cb = SARMetricsCallback(verbose=0)

    # b) Periodic evaluation: runs EVAL_EPISODES complete episodes every
    #    EVAL_FREQ training steps.  Saves the best model so far to disk.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = CHECKPOINT_DIR,
        log_path             = CHECKPOINT_DIR,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),   # per-worker freq
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,   # greedy policy during eval
        render               = False,  # disable rendering during eval for speed
        verbose              = 1,
    )

    # c) Checkpoint saver: writes model_N_steps.zip every CHECKPOINT_FREQ steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_sar",
        verbose     = 1,
    )

    callbacks = [sar_metrics_cb, eval_cb, checkpoint_cb]
    print(f"      ✅  {len(callbacks)} callbacks attached.\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("  Monitor progress:  tensorboard --logdir ppo_sar_tensorboard\n")
    t_start = time.time()

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = callbacks,
        progress_bar    = True,   # requires tqdm: pip install tqdm
        reset_num_timesteps = True,
        tb_log_name     = "PPO_SAR",
    )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed / 60:.1f} min "
          f"({elapsed:.0f} s)")

    # ── Save final model ──────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✅  Final model saved to '{MODEL_PATH}.zip'")

    # ── Cleanup ───────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()

    print("\nTo evaluate, run:  python train_rl.py --mode eval\n")


# ===========================================================================
# SECTION 6  |  EVALUATION FUNCTION
# ===========================================================================

def evaluate(n_episodes: int = N_EVAL_EPISODES) -> None:
    """
    Load the saved PPO model and run *n_episodes* visualised test episodes.

    Each episode plays out in real-time with matplotlib rendering enabled
    so you can observe the learned policy.  Episode-level scores are printed
    at the end of each run and a summary table is printed at the end.

    Parameters
    ----------
    n_episodes : number of test episodes to run (default: N_EVAL_EPISODES = 3)

    Episode scoring (printed to console)
    -------------------------------------
    cumulative_reward  : sum of shaped rewards across the episode
    persons_found      : how many rescue targets were detected (out of 10)
    collisions         : total collision events
    form_dev_mean      : average formation deviation [m]
    path_dev_mean      : average path deviation from sweep lane [m]
    obs_discovered     : how many obstacles were mapped during the run
    """
    # ── Load model ────────────────────────────────────────────────────────
    model_file = f"{MODEL_PATH}.zip"
    if not os.path.exists(model_file):
        print(f"❌  Model file '{model_file}' not found.")
        print("   Run  python train_rl.py --mode train  first.")
        return

    print(f"\nLoading model from '{model_file}'...")
    model = PPO.load(model_file)
    print(f"✅  Model loaded.\n")

    print("=" * 60)
    print(f"  EVALUATION — {n_episodes} test episodes")
    print("=" * 60)

    episode_results = []

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1} / {n_episodes} "
              f"(seed = {SEED + ep}) ---")

        # Fresh env for each episode with a unique seed
        env = SARGymnasiumWrapper(seed=SEED + ep)

        obs, _info     = env.reset()
        cumulative_rew = 0.0
        done           = False
        step_num       = 0

        # ── Rollout ───────────────────────────────────────────────────────
        while not done:
            # Deterministic=True uses the policy mean (no exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            cumulative_rew += reward
            step_num       += 1

            # Render every 3 steps to reduce flicker while staying responsive
            if step_num % 3 == 0:
                env.render()

            done = terminated or truncated

        # ── Episode summary ───────────────────────────────────────────────
        end_reason = ("COLLISION"    if terminated and info['collisions'] > 0
                      else "SUCCESS" if info['found'] >= info['total']
                      else "TIMEOUT")

        result = {
            'episode':         ep + 1,
            'reward':          cumulative_rew,
            'persons_found':   info['found'],
            'persons_total':   info['total'],
            'collisions':      info['collisions'],
            'form_dev_mean':   info['form_dev_mean'],
            'path_dev_mean':   info['path_dev_mean'],
            'obs_discovered':  info['obs_discovered'],
            'steps':           step_num,
            'end_reason':      end_reason,
        }
        episode_results.append(result)

        # Per-episode console output
        print(f"  End condition   : {end_reason}")
        print(f"  Steps taken     : {step_num} / {MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {info['found']} / {info['total']}")
        print(f"  Collisions      : {info['collisions']}")
        print(f"  Form. dev (avg) : {info['form_dev_mean']:.3f} m")
        print(f"  Path  dev (avg) : {info['path_dev_mean']:.3f} m")
        print(f"  Obstacles known : {info['obs_discovered']} / "
              f"{info['obs_total']}")

        # Pause on the final frame so the user can inspect the trail map
        input("  [Press Enter to continue to next episode] ")
        env.close()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Ep':>3}  {'Reward':>9}  {'Found':>8}  "
          f"{'Coll':>5}  {'FormDev':>8}  {'End':>9}")
    print("  " + "-" * 56)
    for r in episode_results:
        found_str = f"{r['persons_found']}/{r['persons_total']}"
        print(f"  {r['episode']:>3}  {r['reward']:>9.1f}  {found_str:>8}  "
              f"{r['collisions']:>5}  {r['form_dev_mean']:>7.3f} m  "
              f"{r['end_reason']:>9}")

    # Aggregate statistics
    rewards  = [r['reward']          for r in episode_results]
    found_f  = [r['persons_found'] / max(r['persons_total'], 1)
                for r in episode_results]
    form_devs = [r['form_dev_mean']  for r in episode_results]

    print("  " + "-" * 56)
    print(f"  Mean reward          : {np.mean(rewards):.1f} "
          f"± {np.std(rewards):.1f}")
    print(f"  Mean rescue rate     : {np.mean(found_f) * 100:.1f}%")
    print(f"  Mean formation dev   : {np.mean(form_devs):.3f} m")
    print("=" * 60)


# ===========================================================================
# SECTION 7  |  ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Modes
    -----
    train  : run the full training pipeline and save the model.
    eval   : load a saved model and run N_EVAL_EPISODES visual test episodes.
    both   : train then immediately evaluate (useful for CI / automated runs).
    check  : run check_env() only (fast smoke-test, no training).
    """
    parser = argparse.ArgumentParser(
        description="PPO training + evaluation for the SAR Swarm environment."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both", "check"],
        default="both",
        help=(
            "train  = train and save model\n"
            "eval   = load saved model and render test episodes\n"
            "both   = train then eval (default)\n"
            "check  = run SB3 env checker only"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EVAL_EPISODES,
        help=f"Number of test episodes for eval mode (default: {N_EVAL_EPISODES})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "check":
        # Quick sanity check — useful before kicking off a long training run
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
