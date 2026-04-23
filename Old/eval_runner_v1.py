from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sar_classical_controller import LeaderCtrl


PLOT_DIR = Path("eval_plots")
DEFAULT_VECSNORM_PATH = Path("checkpoints") / "vecnormalize.pkl"


def _force_canvas_refresh(fig) -> None:
    """Force immediate GUI refresh for interactive Matplotlib backends."""
    if fig is None:
        return
    canvas = getattr(fig, "canvas", None)
    if canvas is None:
        return
    if hasattr(canvas, "draw_idle"):
        canvas.draw_idle()
    if hasattr(canvas, "flush_events"):
        canvas.flush_events()


def _render_wrapper_fast(env: Any, fig, ax):
    """Lighter render path for eval loops (no coverage-cell overlays)."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("#12121f")
        plt.ion()

    leader = env._env.robots[0]
    f1_tgt = None
    f2_tgt = None
    if hasattr(env, '_fctrl1') and hasattr(env, '_fctrl2'):
        f1_tgt = env._fctrl1.formation_target(leader.pose)
        f2_tgt = env._fctrl2.formation_target(leader.pose)

    # Render only active persons to avoid showing disabled slots as pre-found.
    persons_backup = env._env.persons
    found_backup = env._env.total_found
    active_n = int(getattr(env, "_n_active_persons", len(persons_backup)))
    pre_detected = int(getattr(env, "_pre_detected_count", 0))
    active_persons = persons_backup[:active_n]
    active_found = int(np.clip(found_backup - pre_detected, 0, len(active_persons)))
    env._env.persons = active_persons
    env._env.total_found = active_found
    try:
        env._env.render(ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)
    finally:
        env._env.persons = persons_backup
        env._env.total_found = found_backup

    fig.suptitle(
        "SAR Eval [Fast Render] "
        f"t={env._env.t:.1f}s Active={active_n}",
        color="#ccc", fontsize=9,
    )
    _force_canvas_refresh(fig)
    plt.pause(0.001)
    return fig, ax


def _save_comparison_plots(
    rl_rows: list[dict[str, float]],
    classical_rows: list[dict[str, float]],
    seed_base: int,
    n_episodes: int,
) -> tuple[Path, Path]:
    episodes = np.arange(1, len(rl_rows) + 1)

    # 1) Per-episode line charts
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("RL vs Classical: Per-Episode Metrics", fontsize=14)

    metrics = [
        ("form_dev_mean", "Formation Deviation (mean, m)"),
        ("completion_time", "Completion Time (s)"),
        ("persons_found", "People Found"),
        ("reward", "Overall Reward"),
    ]
    better_text = {
        "form_dev_mean": "Lower is better",
        "completion_time": "Lower is better",
        "persons_found": "Higher is better",
        "reward": "Higher is better",
    }

    max_people_total = max(
        [row.get("persons_total", 0.0) for row in rl_rows] +
        [row.get("persons_total", 0.0) for row in classical_rows] +
        [1.0]
    )

    for ax, (key, title) in zip(axes.flat, metrics):
        rl_vals = [row[key] for row in rl_rows]
        cl_vals = [row[key] for row in classical_rows]
        ax.plot(episodes, rl_vals, label="RL", color="#1565c0", linewidth=2)
        ax.plot(episodes, cl_vals, label="Classical", color="#c62828", linewidth=2)
        ax.set_title(f"{title} | {better_text[key]}")
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
        ax.legend()
        if key == "persons_found":
            ax.set_ylim(0.0, float(max_people_total) + 0.5)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    lines_path = PLOT_DIR / f"comparison_lines_seed{seed_base}_{n_episodes}eps.png"
    fig.savefig(lines_path, dpi=160)
    plt.close(fig)

    # 2) Average comparison chart across all runs
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig2.suptitle("RL vs Classical: Average Metrics (Independent Scales)", fontsize=14)

    for ax, (key, title) in zip(axes2.flat, metrics):
        rl_mean = float(np.mean([row[key] for row in rl_rows]))
        cl_mean = float(np.mean([row[key] for row in classical_rows]))
        ax.bar(["RL", "Classical"], [rl_mean, cl_mean], color=["#1565c0", "#c62828"])
        ax.set_ylabel("Average")
        ax.set_title(f"{title} | {better_text[key]}")
        ax.grid(axis="y", alpha=0.3)
        if key == "persons_found":
            ax.set_ylim(0.0, float(max_people_total) + 0.5)

    avgs_path = PLOT_DIR / f"comparison_averages_seed{seed_base}_{n_episodes}eps.png"
    fig2.savefig(avgs_path, dpi=160)
    plt.close(fig2)

    return lines_path, avgs_path


def _open_plot_files(paths: list[Path]) -> None:
    for p in paths:
        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                print(f"Open plot manually: {p}")
        except Exception as exc:
            print(f"Could not auto-open plot '{p}': {exc}")


def load_train_module(train_file: str) -> ModuleType:
    train_path = Path(train_file).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    spec = importlib.util.spec_from_file_location("sar_train_module", str(train_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {train_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_eval_env_like_training(
    env: Any,
    train_mod: ModuleType,
    active_persons: int,
    active_obstacles: int,
) -> None:
    """Apply the same difficulty-related knobs used by training.

    This keeps standalone evaluation behavior aligned with the wrapper
    configuration expected by `train_rl.py`.
    """
    env.set_n_active_persons(active_persons)
    env.set_n_active_obstacles(active_obstacles)

    # Evaluation should remain deterministic/stage-fixed.
    if hasattr(env, "enable_random_difficulty"):
        env.enable_random_difficulty(False)
    if hasattr(env, "set_collision_grace_active"):
        env.set_collision_grace_active(False)

    # Match the stage reward multiplier configured during training callback.
    if hasattr(env, "set_reward_bonus_multiplier"):
        stage_idx = 0
        stages = getattr(train_mod, "CURRICULUM_STAGES", [])
        for i, (_thr, persons, obstacles) in enumerate(stages):
            if int(persons) == int(active_persons) and int(obstacles) == int(active_obstacles):
                stage_idx = i
                break
        bonus_by_stage = getattr(train_mod, "DIFFICULTY_BONUS_BY_STAGE", [1.0])
        if len(bonus_by_stage) == 0:
            bonus = 1.0
        else:
            bonus = float(bonus_by_stage[min(stage_idx, len(bonus_by_stage) - 1)])
        env.set_reward_bonus_multiplier(bonus)


def evaluate_classical_baseline(
    train_mod: ModuleType,
    n_episodes: int,
    seed_base: int,
    active_persons: int,
    active_obstacles: int,
    wait_for_input: bool,
    render_every: int,
    show_final_only: bool,
    final_hold_seconds: float,
) -> dict[str, float]:
    print("\n" + "=" * 65)
    print(f"  CLASSICAL BASELINE EVALUATION - {n_episodes} episodes")
    print("=" * 65)

    rows: list[dict[str, float]] = []
    component_rows: list[dict[str, float]] = []
    render_fig = None
    render_ax = None

    for ep in range(n_episodes):
        print(f"\n--- Classical Episode {ep + 1}/{n_episodes} (seed={seed_base + ep}) ---")

        env = train_mod.SARGymnasiumWrapper(seed=seed_base + ep)
        _configure_eval_env_like_training(env, train_mod, active_persons, active_obstacles)
        obs, _ = env.reset()

        leader_ctrl = LeaderCtrl(env._env.waypoints)

        cumulative_rew = 0.0
        speed_sum = 0.0
        done = False
        step_num = 0
        info: dict[str, Any] = {}

        while not done:
            leader = env._env.robots[0]
            known_obs = env._env.shared_obs.all()
            v_l, w_l = leader_ctrl(leader, known_obs)

            a0 = 2.0 * (float(v_l) / max(train_mod.V_MAX, 1e-9)) - 1.0
            a1 = float(w_l) / max(train_mod.OMEGA_MAX, 1e-9)
            action = np.array([
                np.clip(a0, -1.0, 1.0),
                np.clip(a1, -1.0, 1.0),
            ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_rew += float(reward)
            speed_sum += abs(float(env._env.robots[0].v))
            step_num += 1
            if ((not show_final_only)
                    and step_num % max(render_every, 1) == 0):
                render_fig, render_ax = _render_wrapper_fast(env, render_fig, render_ax)
            done = terminated or truncated

        active_found = float(info.get("active_found", info["found"]))
        active_total = float(info.get("active_total", info["total"]))

        if active_found >= active_total:
            end_reason = "SUCCESS"
        elif info["collisions"] > 0:
            end_reason = "COLLISION"
        else:
            end_reason = "TIMEOUT"

        rows.append(
            {
                "reward": float(cumulative_rew),
                "found": active_found,
                "total": active_total,
                "collisions": float(info["collisions"]),
                "steps": float(step_num),
                "avg_speed": float(speed_sum / max(step_num, 1)),
                "form_dev_mean": float(info["form_dev_mean"]),
                "strict_success": 1.0 if (active_found >= active_total and info["collisions"] == 0) else 0.0,
            }
        )

        print(f"  End condition   : {end_reason}")
        print(f"  Steps           : {step_num} / {train_mod.MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {int(active_found)} / {int(active_total)}")
        print(f"  Collisions      : {info['collisions']}")

        ep_components = dict(info.get("reward_components", {}))
        if ep_components:
            print("  Reward breakdown:")
            for name, val in sorted(ep_components.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(val) > 0.01:
                    print(f"    {name:>20s} : {val:>+9.1f}")

        if show_final_only:
            render_fig, render_ax = _render_wrapper_fast(env, render_fig, render_ax)
            _force_canvas_refresh(render_fig)
            if wait_for_input:
                input("  [Final snapshot shown. Press Enter for next episode] ")
            else:
                plt.pause(max(final_hold_seconds, 0.0))

        component_rows.append(ep_components)
        if wait_for_input and (not show_final_only):
            input("  [Press Enter for next episode] ")
        env.close()

    if render_fig is not None:
        plt.close(render_fig)

    rewards = np.array([r["reward"] for r in rows], dtype=np.float32)
    found_counts = np.array([r["found"] for r in rows], dtype=np.float32)
    total_counts = np.array([r["total"] for r in rows], dtype=np.float32)
    collisions = np.array([r["collisions"] for r in rows], dtype=np.float32)
    strict_success = np.array([r["strict_success"] for r in rows], dtype=np.float32)
    steps = np.array([r["steps"] for r in rows], dtype=np.float32)

    summary = {
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) else 0.0,
        "mean_found": float(np.mean(found_counts)) if len(found_counts) else 0.0,
        "mean_total": float(np.mean(total_counts)) if len(total_counts) else 0.0,
        "mean_collisions": float(np.mean(collisions)) if len(collisions) else 0.0,
        "strict_success_rate": float(np.mean(strict_success)) if len(strict_success) else 0.0,
        "mean_steps": float(np.mean(steps)) if len(steps) else 0.0,
        "rows": rows,
    }

    print("\n" + "=" * 65)
    print("  CLASSICAL BASELINE SUMMARY")
    print("=" * 65)
    print(f"  Mean reward      : {summary['mean_reward']:.1f} +- {summary['std_reward']:.1f}")
    print(f"  Mean found       : {summary['mean_found']:.1f} / {summary['mean_total']:.1f}")
    print(f"  Mean collisions  : {summary['mean_collisions']:.2f}")
    print(f"  Mean steps       : {summary['mean_steps']:.1f}")
    print(f"  Mean time        : {summary['mean_steps'] * train_mod.DT:.1f} s")
    print(f"  Strict success   : {summary['strict_success_rate'] * 100:.1f}%")

    if component_rows:
        print("  Reward breakdown (mean/episode):")
        all_names: list[str] = []
        for row in component_rows:
            for name in row.keys():
                if name not in all_names:
                    all_names.append(name)
        for name in all_names:
            vals = [row.get(name, 0.0) for row in component_rows]
            mean_val = float(np.mean(vals))
            if abs(mean_val) > 0.01:
                print(f"    {name:>20s} : {mean_val:>+9.1f}")

    print("=" * 65)
    return summary


def _normalize_model_prefix(model_path: str) -> str:
    return model_path[:-4] if model_path.lower().endswith(".zip") else model_path


def _resolve_default_model_prefix(train_mod: ModuleType) -> str:
    default_model = train_mod.MODEL_SAVE_PATH
    fallback_model = train_mod.MODEL_PATH
    return default_model if os.path.exists(f"{default_model}.zip") else fallback_model


def _resolve_vecnormalize_path(train_mod: ModuleType) -> Optional[Path]:
    configured = getattr(train_mod, "VECNORM_PATH", None)
    if configured:
        p = Path(str(configured))
        if p.exists():
            return p
    if DEFAULT_VECSNORM_PATH.exists():
        return DEFAULT_VECSNORM_PATH
    return None


def evaluate_rl_model(
    train_mod: ModuleType,
    n_episodes: int,
    device: str,
    wait_for_input: bool,
    model_path: str,
    seed_base: int,
    active_persons: int,
    active_obstacles: int,
    render_every: int,
    show_final_only: bool,
    final_hold_seconds: float,
) -> dict[str, Any]:
    model_prefix = _normalize_model_prefix(model_path)
    model_file = f"{model_prefix}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model '{model_file}' not found.")

    print("\n" + "=" * 65)
    print(f"  RL EVALUATION | model={model_file}")
    print("=" * 65)

    print(f"\nLoading model from '{model_file}'...")
    model = RecurrentPPO.load(model_prefix, device=device)
    expected_obs_dim = int(model.observation_space.shape[0])
    warned_obs_mismatch = False
    vecnorm_path = _resolve_vecnormalize_path(train_mod)

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

    print("OK - Model loaded.\n")
    if vecnorm_path is not None:
        print(f"Using VecNormalize stats: '{vecnorm_path}'")
    else:
        print("[!] VecNormalize stats not found. Falling back to raw observations.")

    obs_normalizer: Optional[VecNormalize] = None
    if vecnorm_path is not None:
        norm_base_env = DummyVecEnv([lambda: train_mod.SARGymnasiumWrapper(seed=seed_base)])
        obs_normalizer = VecNormalize.load(str(vecnorm_path), norm_base_env)
        obs_normalizer.training = False
        obs_normalizer.norm_reward = False

    def _normalize_obs_for_model(raw_obs: np.ndarray) -> np.ndarray:
        obs_vec = np.asarray(raw_obs, dtype=np.float32)
        if obs_normalizer is None:
            return _adapt_obs_dim(obs_vec)
        norm_batch = obs_normalizer.normalize_obs(obs_vec[None, :])
        return _adapt_obs_dim(np.asarray(norm_batch[0], dtype=np.float32))

    print("=" * 65)
    print(
        f"  EVALUATION - {n_episodes} episodes "
        f"({active_persons} persons, {active_obstacles} obstacles)"
    )
    print("=" * 65)

    results = []
    render_fig = None
    render_ax = None
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} (seed={seed_base + ep}) ---")

        raw_env = train_mod.SARGymnasiumWrapper(seed=seed_base + ep)
        _configure_eval_env_like_training(raw_env, train_mod, active_persons, active_obstacles)
        raw_obs, _ = raw_env.reset()
        obs = _normalize_obs_for_model(raw_obs)

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        cumulative_rew = 0.0
        speed_sum = 0.0
        done = False
        step_num = 0
        info: dict[str, Any] = {}

        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts, deterministic=True,
            )
            raw_obs, reward, terminated, truncated, info = raw_env.step(action)
            done = bool(terminated or truncated)
            if not done:
                obs = _normalize_obs_for_model(raw_obs)
            episode_starts = np.zeros((1,), dtype=bool)
            cumulative_rew += float(reward)
            speed_sum += abs(float(raw_env._env.robots[0].v))
            step_num += 1
            if ((not show_final_only)
                    and step_num % max(render_every, 1) == 0):
                render_fig, render_ax = _render_wrapper_fast(raw_env, render_fig, render_ax)

        active_found = float(info.get("active_found", info["found"]))
        active_total = float(info.get("active_total", info["total"]))
        if active_found >= active_total:
            end_reason = "SUCCESS"
        elif info["collisions"] > 0:
            end_reason = "COLLISION"
        else:
            end_reason = "TIMEOUT"

        results.append(
            {
                "episode": ep + 1,
                "reward": float(cumulative_rew),
                "persons_found": active_found,
                "persons_total": active_total,
                "collisions": float(info["collisions"]),
                "form_dev_mean": float(info["form_dev_mean"]),
                "coverage_frac": float(info.get("coverage_frac", 0.0)),
                "steps": float(step_num),
                "avg_speed": float(speed_sum / max(step_num, 1)),
                "end_reason": end_reason,
                "reward_components": dict(info.get("reward_components", {})),
            }
        )

        print(f"  End condition   : {end_reason}")
        print(f"  Steps           : {step_num} / {train_mod.MAX_EPISODE_STEPS}")
        print(f"  Cumul. reward   : {cumulative_rew:.1f}")
        print(f"  Persons found   : {int(active_found)} / {int(active_total)}")
        print(f"  Collisions      : {info['collisions']}")
        print(f"  Form. dev (avg) : {info['form_dev_mean']:.3f} m")
        print(f"  Map coverage    : {info.get('coverage_frac', 0.0) * 100:.1f}%")

        rc = info.get("reward_components", {})
        if rc:
            print("  Reward breakdown:")
            for name, val in sorted(rc.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(val) > 0.01:
                    print(f"    {name:>20s} : {val:>+9.1f}")

        if show_final_only:
            render_fig, render_ax = _render_wrapper_fast(raw_env, render_fig, render_ax)
            _force_canvas_refresh(render_fig)
            if wait_for_input:
                input("  [Final snapshot shown. Press Enter for next episode] ")
            else:
                plt.pause(max(final_hold_seconds, 0.0))

        if wait_for_input and (not show_final_only):
            input("  [Press Enter for next episode] ")
        raw_env.close()

    if obs_normalizer is not None:
        obs_normalizer.close()

    if render_fig is not None:
        plt.close(render_fig)

    rewards = [r["reward"] for r in results]
    found_counts = [r["persons_found"] for r in results]
    total_counts = [r["persons_total"] for r in results]
    coverages = [r["coverage_frac"] for r in results]
    mean_steps = float(np.mean([r["steps"] for r in results])) if results else 0.0
    strict_success_rate = np.mean([
        1.0 if (r["persons_found"] >= r["persons_total"] and r["collisions"] == 0) else 0.0
        for r in results
    ]) if results else 0.0

    summary = {
        "model_path": model_file,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_found": float(np.mean(found_counts)) if found_counts else 0.0,
        "mean_total": float(np.mean(total_counts)) if total_counts else 0.0,
        "mean_collisions": float(np.mean([r["collisions"] for r in results])) if results else 0.0,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_speed": float(np.mean([r["avg_speed"] for r in results])) if results else 0.0,
        "mean_steps": mean_steps,
        "strict_success_rate": float(strict_success_rate),
        "rows": results,
    }

    print("\n" + "=" * 65)
    print("  RL SUMMARY")
    print("=" * 65)
    print(f"  Model            : {model_file}")
    print(f"  Mean reward      : {summary['mean_reward']:.1f} +- {summary['std_reward']:.1f}")
    print(f"  Mean found       : {summary['mean_found']:.1f} / {summary['mean_total']:.1f}")
    print(f"  Mean collisions  : {summary['mean_collisions']:.2f}")
    print(f"  Mean coverage    : {summary['mean_coverage'] * 100:.1f}%")
    print(f"  Mean speed       : {summary['mean_speed']:.3f} m/s")
    print(f"  Mean steps       : {summary['mean_steps']:.1f}")
    print(f"  Mean time        : {summary['mean_steps'] * train_mod.DT:.1f} s")
    print(f"  Strict success   : {summary['strict_success_rate'] * 100:.1f}%")
    print("=" * 65)

    return summary


def run_evaluation(
    train_mod: ModuleType,
    n_episodes: int,
    device: str,
    wait_for_input: bool,
    model_paths: list[str],
    mode: str,
    seed_base: int,
    eval_active_persons: Optional[int],
    eval_active_obstacles: Optional[int],
    render_every: int,
    show_final_only: bool,
    final_hold_seconds: float,
) -> None:
    active_persons = int(
        np.clip(
            train_mod.EVAL_ACTIVE_PERSONS if eval_active_persons is None else eval_active_persons,
            1,
            train_mod.N_PERSONS,
        )
    )
    active_obstacles = int(
        np.clip(
            train_mod.EVAL_ACTIVE_OBSTACLES if eval_active_obstacles is None else eval_active_obstacles,
            0,
            train_mod.N_OBS,
        )
    )

    rl_summaries: list[dict[str, Any]] = []
    if mode in ("rl", "both"):
        selected_models = model_paths[:] if model_paths else [_resolve_default_model_prefix(train_mod)]
        for mp in selected_models:
            rl_summaries.append(
                evaluate_rl_model(
                    train_mod=train_mod,
                    n_episodes=n_episodes,
                    device=device,
                    wait_for_input=wait_for_input,
                    model_path=mp,
                    seed_base=seed_base,
                    active_persons=active_persons,
                    active_obstacles=active_obstacles,
                    render_every=render_every,
                    show_final_only=show_final_only,
                    final_hold_seconds=final_hold_seconds,
                )
            )

    classical: Optional[dict[str, Any]] = None
    if mode in ("classical", "both"):
        classical = evaluate_classical_baseline(
            train_mod=train_mod,
            n_episodes=n_episodes,
            seed_base=seed_base,
            active_persons=active_persons,
            active_obstacles=active_obstacles,
            wait_for_input=wait_for_input,
            render_every=render_every,
            show_final_only=show_final_only,
            final_hold_seconds=final_hold_seconds,
        )

    if mode == "classical":
        return

    if len(rl_summaries) > 1:
        print("\n" + "=" * 90)
        print("  RL MODEL COMPARISON (averages)")
        print("=" * 90)
        print(
            f"  {'Model':<35} {'Reward':>10} {'Found':>12} {'Coll':>8} {'Speed':>8} {'Time(s)':>10}"
        )
        print("  " + "-" * 86)
        for s in rl_summaries:
            print(
                f"  {s['model_path']:<35} {s['mean_reward']:>10.1f} "
                f"{s['mean_found']:>5.1f}/{s['mean_total']:<5.1f} "
                f"{s['mean_collisions']:>8.2f} {s['mean_speed']:>8.3f} "
                f"{(s['mean_steps'] * train_mod.DT):>10.1f}"
            )
        print("=" * 90)

    if classical is not None and rl_summaries:
        for s in rl_summaries:
            rl_mean_reward = s["mean_reward"]
            rl_std_reward = s["std_reward"]
            rl_mean_found = s["mean_found"]
            rl_mean_total = s["mean_total"]
            rl_mean_collisions = s["mean_collisions"]
            rl_mean_steps = s["mean_steps"]

            print("\n" + "=" * 65)
            print(f"  RL VS CLASSICAL (same seeds) | model={s['model_path']}")
            print("=" * 65)
            print(
                "  Reward (mean+-std): "
                f"RL {rl_mean_reward:.1f}+-{rl_std_reward:.1f} | "
                f"Classical {classical['mean_reward']:.1f}+-{classical['std_reward']:.1f}"
            )
            print(
                "  Found (mean)      : "
                f"RL {rl_mean_found:.1f}/{rl_mean_total:.1f} | "
                f"Classical {classical['mean_found']:.1f}/{classical['mean_total']:.1f}"
            )
            print(
                "  Collisions (mean) : "
                f"RL {rl_mean_collisions:.2f} | Classical {classical['mean_collisions']:.2f}"
            )
            print(
                "  Avg time          : "
                f"RL {rl_mean_steps * train_mod.DT:.1f}s ({rl_mean_steps:.1f} steps) | "
                f"Classical {classical['mean_steps'] * train_mod.DT:.1f}s ({classical['mean_steps']:.1f} steps)"
            )
            print(
                "  Strict success    : "
                f"RL {s['strict_success_rate'] * 100:.1f}% | "
                f"Classical {classical['strict_success_rate'] * 100:.1f}%"
            )
            print("=" * 65)

            rl_plot_rows = [
            {
                "reward": float(r["reward"]),
                "form_dev_mean": float(r["form_dev_mean"]),
                "completion_time": float(r["steps"] * train_mod.DT),
                "persons_found": float(r["persons_found"]),
                "persons_total": float(r["persons_total"]),
            }
            for r in s["rows"]
            ]
            classical_plot_rows = [
            {
                "reward": float(r["reward"]),
                "form_dev_mean": float(r["form_dev_mean"]),
                "completion_time": float(r["steps"] * train_mod.DT),
                "persons_found": float(r["found"]),
                "persons_total": float(r["total"]),
            }
            for r in classical.get("rows", [])
            ]
            if len(rl_plot_rows) == len(classical_plot_rows) and rl_plot_rows:
                lines_path, avgs_path = _save_comparison_plots(
                    rl_plot_rows,
                    classical_plot_rows,
                    seed_base=seed_base,
                    n_episodes=n_episodes,
                )
                print(f"Saved line comparison plot: {lines_path}")
                print(f"Saved averages comparison plot: {avgs_path}")
                _open_plot_files([lines_path, avgs_path])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone evaluator for SAR RL models and classical baseline comparison."
    )
    parser.add_argument(
        "--mode", choices=["rl", "classical", "both"], default="both",
        help="rl: RL only | classical: classical only | both: run both",
    )
    parser.add_argument(
        "--train-file",
        default="train_rl.py",
        help="Path to the training script to import (default: train_rl.py)",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"],
        default="auto", help="Torch device for RL model loading",
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Run evaluation without waiting for Enter prompts",
    )
    parser.add_argument(
        "--render-every", type=int, default=30,
        help="Render every N steps in interactive mode (higher = less lag, default: 30)",
    )
    parser.add_argument(
        "--show-final-only", action="store_true",
        help="Skip live rendering and only show a final snapshot at episode end",
    )
    parser.add_argument(
        "--final-hold-seconds", type=float, default=4.0,
        help="When not prompting, keep final snapshot visible for N seconds (default: 4)",
    )
    parser.add_argument(
        "--compare-classical", action="store_true",
        help="Deprecated: use --mode both (kept for backward compatibility)",
    )
    parser.add_argument(
        "--seed-base", "--start-seed", dest="seed_base", type=int, default=None,
        help="Starting seed for episode sequence (default: training SEED)",
    )
    parser.add_argument(
        "--eval-persons", type=int, default=None,
        help="Active persons during eval (default: training file eval default)",
    )
    parser.add_argument(
        "--eval-obstacles", type=int, default=None,
        help="Active obstacles during eval (default: training file eval default)",
    )
    parser.add_argument(
        "--model-path", action="append", default=None,
        help="RL model path (repeat flag to compare multiple RL models); accepts with or without .zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mod = load_train_module(args.train_file)

    seed_base = train_mod.SEED if args.seed_base is None else int(args.seed_base)
    mode = args.mode
    if args.compare_classical and mode == "rl":
        mode = "both"

    model_paths = args.model_path if args.model_path is not None else []

    run_evaluation(
        train_mod=train_mod,
        n_episodes=int(args.episodes),
        device=args.device,
        wait_for_input=not args.no_prompt,
        model_paths=model_paths,
        mode=mode,
        seed_base=seed_base,
        eval_active_persons=args.eval_persons,
        eval_active_obstacles=args.eval_obstacles,
        render_every=max(int(args.render_every), 1),
        show_final_only=bool(args.show_final_only),
        final_hold_seconds=max(float(args.final_hold_seconds), 0.0),
    )


if __name__ == "__main__":
    main()
