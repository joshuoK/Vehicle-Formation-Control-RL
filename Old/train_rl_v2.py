"""
train_rl_v2.py
==============
PPO training script for the SAR Swarm environment.

Changes vs original
-------------------
1.  HYBRID CONTROL
    Followers driven by classical APF. RL only learns 2-D leader policy.

2.  SEED DIVERSITY
    Episode seed increments each reset. Workers offset by rank*1000.

3.  FIXED TERMINATION
    Collisions penalised (-40) but do NOT end the episode.
    Only mission success or step limit end an episode.

4.  REWARD OVERHAUL
      +50  per person found
      -40  per new collision
      +0.2 per step leader moves forward
      +1.0 when form_dev < 0.5 m
      -0.02 per step (time pressure)

Quick-start
-----------
    pip install stable-baselines3[extra] gymnasium matplotlib numpy tqdm
    python train_rl_v2.py --mode train
    python train_rl_v2.py --mode eval
    python train_rl_v2.py --mode both
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

from stable_baselines3 import PPO
from stable_baselines3.common.env_util     import make_vec_env
from stable_baselines3.common.vec_env      import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks    import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.monitor      import Monitor
from stable_baselines3.common.env_checker  import check_env

sys.path.insert(0, os.path.dirname(__file__))
from sar_environment import (
    SAREnvironment, V_MAX, OMEGA_MAX, N_PERSONS, FORM_OFFSET,
)
from sar_classical_controller import APFFollowerCtrl

multiprocessing.freeze_support()
_IS_WINDOWS      = (platform.system() == 'Windows')
_MP_START_METHOD = 'spawn' if _IS_WINDOWS else 'fork'


# ===========================================================================
# SECTION 1  |  HYPERPARAMETERS
# ===========================================================================

N_ENVS            = 4
MAX_EPISODE_STEPS = 4000
SEED              = 2024

R_PERSON_FOUND    =  50.0
R_COLLISION       = -40.0
R_FORWARD         =   0.2
R_FORMATION_BONUS =   1.0
R_TIME_PENALTY    =  -0.02
R_PERSON_MISSED   = -200.0  # per person NOT found when episode ends

LEARNING_RATE  = 3e-4
N_STEPS        = 2048
BATCH_SIZE     = 256
N_EPOCHS       = 10
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.01
VF_COEF        = 0.5
MAX_GRAD_NORM  = 0.5

POLICY_KWARGS = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

TOTAL_TIMESTEPS  = 1_500_000
CHECKPOINT_FREQ  = 50_000
EVAL_FREQ        = 20_000
EVAL_EPISODES    = 5
N_EVAL_EPISODES  = 3

MODEL_PATH      = "ppo_swarm_agent_v2"
TENSORBOARD_LOG = "./ppo_sar_tensorboard/"
CHECKPOINT_DIR  = "./checkpoints_v2/"


# ===========================================================================
# SECTION 2  |  GYMNASIUM WRAPPER
# ===========================================================================

class SARGymnasiumWrapper(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._base_seed     = seed
        self._episode_count = 0

        self._env    = SAREnvironment(seed=seed)
        self._fctrl1 = APFFollowerCtrl(FORM_OFFSET[1])
        self._fctrl2 = APFFollowerCtrl(FORM_OFFSET[2])

        obs_dim = SAREnvironment.OBS_DIM
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._step_count:      int = 0
        self._prev_found:      int = 0
        self._prev_collisions: int = 0

    def _rescale_leader(self, action: np.ndarray) -> tuple:
        a   = np.clip(action, -1.0, 1.0)
        v_l = float((a[0] + 1.0) / 2.0 * V_MAX)
        w_l = float(a[1] * OMEGA_MAX)
        return v_l, w_l

    def _follower_actions(self) -> tuple:
        leader, f1, f2 = self._env.robots
        known_obs = self._env.shared_obs.all()
        v_f1, w_f1, _ = self._fctrl1(f1, leader.pose, known_obs)
        v_f2, w_f2, _ = self._fctrl2(f2, leader.pose, known_obs)
        return v_f1, w_f1, v_f2, w_f2

    def _shape_reward(self, info: dict, raw_obs: np.ndarray,
                      episode_over: bool = False) -> float:
        reward = 0.0

        current_found  = info['found']
        new_detections = current_found - self._prev_found
        if new_detections > 0:
            reward += R_PERSON_FOUND * new_detections
        self._prev_found = current_found

        if info['collisions'] > self._prev_collisions:
            reward += R_COLLISION
        self._prev_collisions = info['collisions']

        if float(raw_obs[3]) > 0.05:
            reward += R_FORWARD

        if info['form_dev_mean'] < 0.5:
            reward += R_FORMATION_BONUS

        reward += R_TIME_PENALTY

        # End-of-episode penalty for missed persons
        if episode_over:
            missed = info['total'] - info['found']
            if missed > 0:
                reward += R_PERSON_MISSED * missed

        return float(reward)

    def reset(self, *, seed=None, options=None) -> tuple:
        if seed is not None:
            self._env.seed = seed
        else:
            self._env.seed = self._base_seed + self._episode_count
        self._episode_count   += 1
        obs                    = self._env.reset()
        self._step_count       = 0
        self._prev_found       = 0
        self._prev_collisions  = 0
        return obs, {}

    def step(self, action: np.ndarray) -> tuple:
        v_l, w_l = self._rescale_leader(action)
        v_f1, w_f1, v_f2, w_f2 = self._follower_actions()
        raw_obs, _base_reward, _done, info = self._env.step(
            [v_l, w_l, v_f1, w_f1, v_f2, w_f2]
        )
        self._step_count += 1
        terminated = int(info['found']) >= int(info['total'])
        truncated  = (not terminated) and (self._step_count >= MAX_EPISODE_STEPS)
        episode_over = terminated or truncated
        reward = self._shape_reward(info, raw_obs, episode_over=episode_over)
        return raw_obs, reward, terminated, truncated, info

    def render(self) -> None:
        if not hasattr(self, '_fig'):
            self._fig, self._ax = plt.subplots(figsize=(9, 9))
            self._fig.patch.set_facecolor('#12121f')
            plt.ion()
        leader = self._env.robots[0]
        f1_tgt = self._fctrl1.formation_target(leader.pose)
        f2_tgt = self._fctrl2.formation_target(leader.pose)
        self._env.render(self._ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)
        self._fig.suptitle(
            f'SAR Swarm [RL+APF v2]  t={self._env.t:.1f}s',
            color='#ccc', fontsize=10,
        )
        plt.pause(0.001)

    def close(self) -> None:
        if hasattr(self, '_fig'):
            plt.close(self._fig)


# ===========================================================================
# SECTION 3  |  FACTORY + CALLBACKS
# ===========================================================================

def make_env(rank: int, base_seed: int = SEED):
    def _init() -> gym.Env:
        return Monitor(SARGymnasiumWrapper(seed=base_seed + rank * 1000))
    return _init


class SARMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get('infos', []),
            self.locals.get('dones', []),
        ):
            if done and 'found' in info:
                total = max(info.get('total', 1), 1)
                obs_t = max(info.get('obs_total', 1), 1)
                self.logger.record('sar/persons_found', info['found'] / total)
                self.logger.record('sar/collisions', float(info['collisions']))
                self.logger.record('sar/form_dev_mean', info['form_dev_mean'])
                self.logger.record('sar/path_dev_mean', info['path_dev_mean'])
                self.logger.record('sar/obs_discovered',
                                   info['obs_discovered'] / obs_t)
        return True


# ===========================================================================
# SECTION 4  |  TRAIN / EVAL
# ===========================================================================

def train() -> None:
    print("=" * 60)
    print("  SAR SWARM  —  PPO  v2  (v1 + missing-persons penalty)")
    print("=" * 60)

    print("\n[1/5] Checking environment wrapper...")
    _check = SARGymnasiumWrapper(seed=SEED)
    check_env(_check, warn=True)
    _check.close()
    print("      ✅  OK.\n")

    print("[2/5] Spawning training environments...")
    train_env = SubprocVecEnv(
        [make_env(i, SEED) for i in range(N_ENVS)],
        start_method=_MP_START_METHOD,
    )
    train_env = VecMonitor(train_env)
    print(f"      ✅  {N_ENVS} workers ready.\n")

    print("[3/5] Building eval environment...")
    eval_env = make_vec_env(
        lambda: Monitor(SARGymnasiumWrapper(seed=SEED + 99999)), n_envs=1,
    )
    print("      ✅  Ready.\n")

    print("[4/5] Instantiating PPO...")
    model = PPO(
        policy="MlpPolicy", env=train_env,
        learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENT_COEF, vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM, policy_kwargs=POLICY_KWARGS,
        tensorboard_log=TENSORBOARD_LOG, verbose=1, seed=SEED,
    )
    print(f"      ✅  Built.\n")

    print("[5/5] Configuring callbacks...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        SARMetricsCallback(),
        EvalCallback(
            eval_env, best_model_save_path=CHECKPOINT_DIR,
            log_path=CHECKPOINT_DIR,
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=EVAL_EPISODES, deterministic=True,
            render=False, verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=CHECKPOINT_DIR, name_prefix="ppo_sar_v2", verbose=1,
        ),
    ]
    print(f"      ✅  {len(callbacks)} callbacks.\n")

    print(f"Training for {TOTAL_TIMESTEPS:,} steps...")
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
        progress_bar=True, reset_num_timesteps=True, tb_log_name="PPO_SAR_v2",
    )
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    model.save(MODEL_PATH)
    print(f"✅  Saved to '{MODEL_PATH}.zip'")
    train_env.close()
    eval_env.close()


def evaluate(n_episodes: int = N_EVAL_EPISODES) -> None:
    model_file = f"{MODEL_PATH}.zip"
    if not os.path.exists(model_file):
        print(f"❌  '{model_file}' not found. Run --mode train first.")
        return
    model = PPO.load(model_file)
    print(f"✅  Model loaded.\n")
    results = []

    for ep in range(n_episodes):
        env = SARGymnasiumWrapper(seed=SEED + ep)
        obs, _ = env.reset()
        cumrew, done, step_num = 0.0, False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            cumrew   += reward
            step_num += 1
            if step_num % 3 == 0:
                env.render()
            done = terminated or truncated

        end = "SUCCESS" if info['found'] >= info['total'] else "TIMEOUT"
        results.append({
            'ep': ep+1, 'reward': cumrew,
            'found': info['found'], 'total': info['total'],
            'collisions': info['collisions'],
            'form_dev': info['form_dev_mean'], 'end': end,
        })
        print(f"Ep {ep+1}: {end}  found={info['found']}/{info['total']}  "
              f"reward={cumrew:.1f}  collisions={info['collisions']}")
        input("  [Enter] ")
        env.close()

    rewards = [r['reward'] for r in results]
    found_f = [r['found']/max(r['total'],1) for r in results]
    print(f"\nMean reward: {np.mean(rewards):.1f}  "
          f"Mean rescue rate: {np.mean(found_f)*100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","eval","both","check"],
                   default="both")
    p.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "check":
        env = SARGymnasiumWrapper(seed=SEED)
        check_env(env, warn=True)
        env.close()
        print("✅  check_env() passed.")
    elif args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate(args.episodes)
    elif args.mode == "both":
        train()
        evaluate(args.episodes)
