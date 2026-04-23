"""
eval_runner.py — SAR Swarm Evaluator (RL + Classical + 3D MP4 export)

Unified evaluation script.  Works with any training module (v9, v10, …).

Modes
─────
  rl        — run RL model only
  classical — run classical boustrophedon baseline only
  both      — run both and compare (default)

3D video
────────
  After evaluation the best RL episode is rendered as a 3D animated MP4
  showing robots, coverage, persons, and obstacles.
  Requires FFmpeg in PATH.  Fallback: Pillow (GIF).
    Install: conda install -c conda-forge ffmpeg
         or: pip install imageio-ffmpeg

Usage examples
──────────────
  python eval_runner.py                                    # both modes, 5 eps
  python eval_runner.py --train-file train_rl_v10.py      # explicit module
  python eval_runner.py --mode rl --episodes 20           # RL only, 20 eps
  python eval_runner.py --no-video                        # skip MP4 export
  python eval_runner.py --all-episodes                    # MP4 for every ep
  python eval_runner.py --model ppo_swarm_agent_v10       # specific model
  python eval_runner.py --rotate --elev 40 --azim -45     # camera options
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive: safe for both 2D plots and 3D MP4
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

sys.path.insert(0, str(Path(__file__).parent))
from sar_classical_controller import LeaderCtrl
from sar_environment import MAP_W, MAP_H, DT, FOV_ANG, FOV_RANGE, SENSOR_OFF


# ===========================================================================
# SECTION 1 | CONSTANTS
# ===========================================================================

PLOT_DIR    = Path("eval_plots")
VIDEO_DIR   = Path("eval_videos")
VIDEO_FPS   = 24
FRAME_SKIP  = 5      # record every Nth env step for the MP4
ROBOT_Z     = 0.45   # robot height above floor in 3D scene [m]
_HALF_FOV   = float(0.5 * FOV_ANG)

ROBOT_COLORS = ["#4fc3f7", "#ef5350", "#ffa726"]   # leader, f1, f2
ROBOT_LABELS = ["Leader", "Follower-1", "Follower-2"]
PERSON_COLOR_UNDET = "#ffee58"
PERSON_COLOR_DET   = "#00e676"
OBS_COLOR_KNOWN    = "#4a4a8c"
OBS_COLOR_HIDDEN   = "#2a2a4c"
COVER_COLOR        = "#00e676"
BG_2D              = "#12121f"
BG_3D              = "#0a0a14"
TRAIL_ALPHA        = 0.45
COVER_ALPHA        = 0.90
OBSTACLE_HEIGHT_SCALE = 2.0


# ===========================================================================
# SECTION 2 | EPISODE RECORDER
# ===========================================================================

class EpisodeRecorder:
    """Snapshot environment state every FRAME_SKIP steps for 3D animation."""

    __slots__ = ("frames", "_step", "_active_persons")

    def __init__(self, active_persons: int) -> None:
        self.frames: list[dict[str, Any]] = []
        self._step: int = 0
        self._active_persons = active_persons

    def record(self, env: Any, info: dict) -> None:
        self._step += 1
        if self._step % FRAME_SKIP != 0:
            return

        env_inner = env._env
        robots = [(float(r.x), float(r.y), float(r.theta)) for r in env_inner.robots]
        trails = [list(r.trail) for r in env_inner.robots]
        persons = [
            (float(p.x), float(p.y), bool(p.detected))
            for p in env_inner.persons[:self._active_persons]
        ]
        known_keys = set(env_inner.shared_obs._cells.keys())
        n_obs = getattr(env, "_n_active_obstacles",
                        getattr(env._env, "_n_active_obstacles",
                                len(env_inner.true_obstacles)))
        obstacles = [
            (float(ox), float(oy), float(r), env_inner.shared_obs._key(ox, oy) in known_keys)
            for ox, oy, r in env_inner.true_obstacles[:n_obs]
        ]
        coverage = getattr(env, "_coverage_grid", None)
        self.frames.append({
            "step":          self._step,
            "robots":        robots,
            "trails":        trails,
            "persons":       persons,
            "obstacles":     obstacles,
            "coverage_grid": coverage.copy() if coverage is not None else None,
            "found":         info.get("active_found", info.get("found", 0)),
            "coverage_frac": info.get("coverage_frac", 0.0),
        })

    def reset(self) -> None:
        self.frames.clear()
        self._step = 0


# ===========================================================================
# SECTION 3 | 3D RENDERING
# ===========================================================================

def _draw_coverage_3d(ax, coverage_grid: np.ndarray, grid_n: int) -> None:
    """Draw covered cells as a floor surface to avoid z-sort artifacts."""
    import matplotlib.colors as mcolors

    if coverage_grid is None:
        return

    coverage_2d = coverage_grid.reshape(grid_n, grid_n)
    rgba_on  = np.array(mcolors.to_rgba(COVER_COLOR, alpha=COVER_ALPHA))
    rgba_off = np.array([0.0, 0.0, 0.0, 0.0])
    facecolors = np.where(coverage_2d.T[:, :, np.newaxis], rgba_on, rgba_off)

    x = np.linspace(0, MAP_W, grid_n + 1)
    y = np.linspace(0, MAP_H, grid_n + 1)
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(
        X, Y, np.zeros_like(X),
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def _draw_cylinder(ax, ox: float, oy: float, rad: float, is_known: bool) -> None:
    """Draw one obstacle as a solid 3D cylinder."""
    color  = OBS_COLOR_KNOWN if is_known else OBS_COLOR_HIDDEN
    alpha  = 0.85     if is_known else 0.35
    height = rad * OBSTACLE_HEIGHT_SCALE
    t      = np.linspace(0, 2 * np.pi, 48)
    z_vals = np.linspace(0.0, height, 6)
    T, Z   = np.meshgrid(t, z_vals)
    ax.plot_surface(ox + rad * np.cos(T), oy + rad * np.sin(T), Z,
                    color=color, alpha=alpha, linewidth=0, antialiased=True)
    # Top cap
    r_vals = np.linspace(0, rad, 8)
    T2, R2 = np.meshgrid(t, r_vals)
    ax.plot_surface(ox + R2 * np.cos(T2), oy + R2 * np.sin(T2),
                    np.full_like(T2, height),
                    color=color, alpha=alpha, linewidth=0, antialiased=True)


def _draw_fov_arc(ax, rx: float, ry: float, rtheta: float,
                  s_off: float, color: str) -> None:
    hdg = rtheta + s_off
    arc = np.linspace(hdg - _HALF_FOV, hdg + _HALF_FOV, 20)
    xs  = [rx] + (rx + FOV_RANGE * np.cos(arc)).tolist() + [rx]
    ys  = [ry] + (ry + FOV_RANGE * np.sin(arc)).tolist() + [ry]
    ax.plot3D(xs, ys, [0.25] * len(xs), color=color, alpha=0.35, lw=1.0)


def _draw_robot_3d(ax, rx: float, ry: float, rtheta: float, color: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    c, s = np.cos(rtheta), np.sin(rtheta)

    def w(lx: float, ly: float, lz: float):
        return (rx + lx * c - ly * s,
                ry + lx * s + ly * c,
                lz)

    def box(lx0, lx1, ly0, ly1, lz0, lz1):
        v = [w(lx0,ly0,lz0), w(lx1,ly0,lz0), w(lx1,ly1,lz0), w(lx0,ly1,lz0),
             w(lx0,ly0,lz1), w(lx1,ly0,lz1), w(lx1,ly1,lz1), w(lx0,ly1,lz1)]
        return [
            [v[4],v[5],v[6],v[7]],
            [v[0],v[1],v[2],v[3]],
            [v[0],v[1],v[5],v[4]],
            [v[3],v[2],v[6],v[7]],
            [v[1],v[2],v[6],v[5]],
            [v[0],v[3],v[7],v[4]],
        ]

    ax.add_collection3d(Poly3DCollection(
        box(-0.40, 0.40, -0.22, 0.22, 0.01, 0.14),
        facecolor=color, edgecolor="#ffffff", alpha=0.92, linewidths=0.3,
    ))
    for ty0, ty1 in [(-0.34, -0.24), (0.24, 0.34)]:
        ax.add_collection3d(Poly3DCollection(
            box(-0.46, 0.46, ty0, ty1, 0.0, 0.09),
            facecolor="#1c1c2e", edgecolor="none", alpha=0.95,
        ))
    ax.add_collection3d(Poly3DCollection(
        box(0.05, 0.28, -0.10, 0.10, 0.14, 0.30),
        facecolor=color, edgecolor="#ffffff", alpha=0.92, linewidths=0.3,
    ))

    ax.plot3D([rx, rx + 0.7 * c], [ry, ry + 0.7 * s], [0.22, 0.22],
              color=color, lw=1.8)
    ax.plot3D([rx, rx], [ry, ry], [0, ROBOT_Z],
              color=color, alpha=0.22, lw=0.8, ls="--")


def render_frame_3d(ax, frame: dict, episode_num: int,
                    grid_n: int, active_persons: int) -> None:
    """Redraw the complete 3D scene for one recorded frame."""
    ax.cla()
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(0, MAP_H)
    ax.set_zlim(0, 5.0)
    ax.set_facecolor(BG_3D)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e1e35")
    ax.set_xlabel("X [m]", color="#666", fontsize=7, labelpad=2)
    ax.set_ylabel("Y [m]", color="#666", fontsize=7, labelpad=2)
    ax.set_zlabel("",       color="#666", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)

    # Floor outline + grid
    bx = [0, MAP_W, MAP_W, 0, 0]
    by = [0, 0, MAP_H, MAP_H, 0]
    ax.plot3D(bx, by, [0]*5, color="#334", lw=1.0)
    for v in np.arange(0, MAP_W + 1, 5):
        ax.plot3D([v, v], [0, MAP_H], [0, 0], color="#1a1a2e", lw=0.4)
    for h in np.arange(0, MAP_H + 1, 5):
        ax.plot3D([0, MAP_W], [h, h], [0, 0], color="#1a1a2e", lw=0.4)

    # Coverage intentionally hidden for cleaner visualization.

    # Obstacles
    for ox, oy, rad, is_known in frame["obstacles"]:
        _draw_cylinder(ax, ox, oy, rad, is_known)

    # Persons
    for px, py, detected in frame["persons"]:
        if detected:
            ax.scatter3D([px], [py], [0.15], s=260, c=PERSON_COLOR_DET,
                         marker="*", depthshade=False, zorder=9)
            ring = np.linspace(0, 2 * np.pi, 28)
            ax.plot3D(px + 0.6 * np.cos(ring), py + 0.6 * np.sin(ring),
                      [0.02] * 28, color=PERSON_COLOR_DET, alpha=0.5, lw=0.8)
        else:
            ax.scatter3D([px], [py], [0.1], s=180, c=PERSON_COLOR_UNDET,
                         marker="P", depthshade=False, zorder=8)

    # FOV arcs, trails, robot bodies
    for i, ((rx, ry, rtheta), color, trail) in enumerate(
        zip(frame["robots"], ROBOT_COLORS, frame["trails"])
    ):
        _draw_fov_arc(ax, rx, ry, rtheta, float(SENSOR_OFF[i]), color)
        if len(trail) > 2:
            ta = np.asarray(trail[-500:])
            ax.plot3D(ta[:, 0], ta[:, 1], np.full(len(ta), 0.05),
                      color=color, alpha=TRAIL_ALPHA, lw=0.8)
        _draw_robot_3d(ax, rx, ry, rtheta, color)

    # Title HUD
    ax.set_title(
        f"Episode {episode_num}  |  t = {frame['step'] * DT:.1f} s  "
        f"found = {int(frame['found'])}/{active_persons}  "
        f"coverage = {frame['coverage_frac']*100:.0f}%",
        color="white", fontsize=9, pad=8,
    )

    # Legend
    leg = (
        [plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=c, markersize=7, label=lbl, linestyle="None")
         for c, lbl in zip(ROBOT_COLORS, ROBOT_LABELS)]
        + [
            plt.Line2D([0], [0], marker="P", color="w",
                       markerfacecolor=PERSON_COLOR_UNDET, markersize=8,
                       label="Person (undetected)", linestyle="None"),
            plt.Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=PERSON_COLOR_DET, markersize=10,
                       label="Person (found)", linestyle="None"),
        ]
    )
    ax.legend(handles=leg, loc="upper left", fontsize=6,
              facecolor="#0e0e20", edgecolor="#4fc3f7",
              labelcolor="white", framealpha=0.7)


# ===========================================================================
# SECTION 4 | MP4 EXPORT
# ===========================================================================

def save_episode_mp4(
    frames: list[dict],
    output_path: Path,
    episode_num: int,
    train_mod: ModuleType,
    active_persons: int,
    elev: float = 35.0,
    azim: float = -55.0,
    rotate: bool = False,
) -> None:
    """Animate recorded frames and write to MP4 (or GIF fallback)."""
    if not frames:
        print("  [!] No frames recorded — skipping video export.")
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    print(f"\n  Rendering {len(frames)} frames → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_n = int(getattr(train_mod, "GRID_N", 20))
    pbar_w = 40

    fig = plt.figure(figsize=(16, 9), facecolor=BG_3D)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    def _update(i: int) -> None:
        render_frame_3d(ax, frames[i], episode_num,
                        grid_n, active_persons)
        if rotate:
            ax.view_init(elev=elev, azim=azim + i * 180.0 / max(len(frames), 1))
        pct  = (i + 1) / len(frames)
        done = int(pbar_w * pct)
        sys.stdout.write(f"\r  [{'#'*done}{'-'*(pbar_w-done)}] {pct*100:.0f}%")
        sys.stdout.flush()

    animation_obj = anim_mod.FuncAnimation(
        fig, _update, frames=len(frames),
        interval=1000 // VIDEO_FPS, repeat=False, blit=False,
    )

    # Try FFMpeg → Pillow → warn
    for writer_cls, path, kwargs in [
        (anim_mod.FFMpegWriter,
         output_path,
         dict(fps=VIDEO_FPS, bitrate=2500, codec="libx264",
              metadata={"title": "SAR Eval"},
              extra_args=["-pix_fmt", "yuv420p"])),
        (anim_mod.PillowWriter,
         output_path.with_suffix(".gif"),
         dict(fps=VIDEO_FPS)),
    ]:
        try:
            w = writer_cls(**kwargs)
            animation_obj.save(str(path), writer=w, dpi=120)
            print(f"\n  Video saved → {path}")
            break
        except Exception as exc:
            print(f"\n  [{writer_cls.__name__}] failed: {exc}")
    else:
        print("  [!] Could not save video. Install FFmpeg or Pillow.")

    plt.close(fig)


# ===========================================================================
# SECTION 5 | 2D LIVE PREVIEW
# ===========================================================================

def _render_2d_live(env: Any, fig, ax):
    """Fast 2D render for interactive watching (no coverage overlay)."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor(BG_2D)
        plt.ion()

    leader = env._env.robots[0]
    f1_tgt = f2_tgt = None
    if hasattr(env, "_fctrl1"):
        f1_tgt = env._fctrl1.formation_target(leader.pose)
        f2_tgt = env._fctrl2.formation_target(leader.pose)

    # Temporarily swap out inactive persons so render shows correct counts
    p_bak, f_bak = env._env.persons, env._env.total_found
    n_active  = int(getattr(env, "_n_active_persons", T_ACTIVE(env)))
    pre_det   = int(getattr(env, "_pre_detected_count", 0))
    env._env.persons     = p_bak[:n_active]
    env._env.total_found = int(np.clip(f_bak - pre_det, 0, n_active))
    try:
        env._env.render(ax, f1_tgt=f1_tgt, f2_tgt=f2_tgt, wp_idx=0)
    finally:
        env._env.persons     = p_bak
        env._env.total_found = f_bak

    fig.suptitle(f"SAR Eval  t={env._env.t:.1f}s  active={n_active}",
                 color="#ccc", fontsize=9)

    canvas = fig.canvas
    if hasattr(canvas, "draw_idle"):
        canvas.draw_idle()
    if hasattr(canvas, "flush_events"):
        canvas.flush_events()
    plt.pause(0.001)
    return fig, ax


def T_ACTIVE(env: Any) -> int:
    """Fallback: total number of persons in the env."""
    return len(env._env.persons)


# ===========================================================================
# SECTION 6 | COMPARISON PLOTS
# ===========================================================================

def _save_comparison_plots(rl_rows: list[dict], classical_rows: list[dict],
                            seed_base: int, n_episodes: int,
                            ) -> tuple[Path, Path]:
    episodes = np.arange(1, len(rl_rows) + 1)
    max_ppl  = max(
        max((r.get("persons_total", 0) for r in rl_rows), default=0),
        max((r.get("persons_total", 0) for r in classical_rows), default=0),
        1.0,
    )
    metrics = [
        ("form_dev_mean",   "Formation Deviation (mean, m)", "Lower is better"),
        ("completion_time", "Completion Time (s)",            "Lower is better"),
        ("persons_found",   "People Found",                   "Higher is better"),
        ("reward",          "Overall Reward",                 "Higher is better"),
    ]

    # Per-episode line chart
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("RL vs Classical — Per-Episode Metrics", fontsize=14)
    for ax, (key, title, note) in zip(axes.flat, metrics):
        ax.plot(episodes, [r[key] for r in rl_rows],
                label="RL", color="#1565c0", lw=2)
        ax.plot(episodes, [r[key] for r in classical_rows],
                label="Classical", color="#c62828", lw=2)
        ax.set_title(f"{title} | {note}")
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
        ax.legend()
        if key == "persons_found":
            ax.set_ylim(0.0, float(max_ppl) + 0.5)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    lines_path = PLOT_DIR / f"comparison_lines_seed{seed_base}_{n_episodes}eps.png"
    fig.savefig(lines_path, dpi=150)
    plt.close(fig)

    # Average bar chart
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig2.suptitle("RL vs Classical — Average Metrics", fontsize=14)
    for ax, (key, title, note) in zip(axes2.flat, metrics):
        ax.bar(["RL", "Classical"],
               [np.mean([r[key] for r in rl_rows]),
                np.mean([r[key] for r in classical_rows])],
               color=["#1565c0", "#c62828"])
        ax.set_ylabel("Average")
        ax.set_title(f"{title} | {note}")
        ax.grid(axis="y", alpha=0.3)
        if key == "persons_found":
            ax.set_ylim(0.0, float(max_ppl) + 0.5)
    avgs_path = PLOT_DIR / f"comparison_averages_seed{seed_base}_{n_episodes}eps.png"
    fig2.savefig(avgs_path, dpi=150)
    plt.close(fig2)

    return lines_path, avgs_path


# ===========================================================================
# SECTION 7 | MODULE + ENV HELPERS
# ===========================================================================

def load_train_module(train_file: str) -> ModuleType:
    path = Path(train_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    spec = importlib.util.spec_from_file_location("sar_train_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_active_counts(train_mod: ModuleType,
                       override_persons: Optional[int],
                       override_obstacles: Optional[int],
                       ) -> tuple[int, int]:
    """
    Resolve active persons/obstacles for eval.
    Supports both v9-style (EVAL_ACTIVE_PERSONS) and v10-style (ACTIVE_PERSONS).
    """
    n_ppl = int(np.clip(
        override_persons if override_persons is not None else
        getattr(train_mod, "ACTIVE_PERSONS",
                getattr(train_mod, "EVAL_ACTIVE_PERSONS", 3)),
        1, train_mod.N_PERSONS,
    ))
    n_obs = int(np.clip(
        override_obstacles if override_obstacles is not None else
        getattr(train_mod, "ACTIVE_OBSTACLES",
                getattr(train_mod, "EVAL_ACTIVE_OBSTACLES", 2)),
        0, train_mod.N_OBS,
    ))
    return n_ppl, n_obs


def _configure_env(env: Any, train_mod: ModuleType,
                   n_persons: int, n_obstacles: int) -> None:
    """Apply difficulty settings to a bare wrapper env."""
    env.set_n_active_persons(n_persons)
    env.set_n_active_obstacles(n_obstacles)
    if hasattr(env, "enable_random_difficulty"):
        env.enable_random_difficulty(False)
    if hasattr(env, "set_collision_grace_active"):
        env.set_collision_grace_active(False)
    if hasattr(env, "set_reward_bonus_multiplier"):
        # Find matching stage bonus (v9) or default to 1.0 (v10)
        stages   = getattr(train_mod, "CURRICULUM_STAGES", [])
        bonuses  = getattr(train_mod, "DIFFICULTY_BONUS_BY_STAGE", [1.0])
        idx      = next(
            (i for i, (_, p, o) in enumerate(stages)
             if int(p) == n_persons and int(o) == n_obstacles),
            0,
        )
        env.set_reward_bonus_multiplier(
            float(bonuses[min(idx, len(bonuses) - 1)]) if bonuses else 1.0
        )


# ===========================================================================
# SECTION 8 | OBS NORMALISATION
# ===========================================================================

def _load_vecnorm(train_mod: ModuleType, seed: int) -> Optional[VecNormalize]:
    """Load VecNormalize stats; return None if file not found."""
    paths = [
        getattr(train_mod, "VECNORM_PATH", None),
        Path("checkpoints") / "vecnormalize.pkl",
        Path("checkpoints_v10") / "vecnormalize.pkl",
    ]
    for p in paths:
        if p and Path(str(p)).exists():
            dummy = DummyVecEnv([lambda: train_mod.SARGymnasiumWrapper(seed=seed)])
            vn = VecNormalize.load(str(p), dummy)
            vn.training    = False
            vn.norm_reward = False
            print(f"  VecNormalize loaded from '{p}'")
            return vn
    print("  [!] VecNormalize not found — using raw observations.")
    return None


def _norm_obs(raw_obs: np.ndarray, vecnorm: Optional[VecNormalize],
              expected_dim: int) -> np.ndarray:
    """Normalise and dimension-adapt a single observation vector."""
    obs = np.asarray(raw_obs, dtype=np.float32)
    if vecnorm is not None:
        obs = np.asarray(vecnorm.normalize_obs(obs[None, :])[0], dtype=np.float32)
    d = obs.shape[0]
    if d > expected_dim:
        obs = obs[:expected_dim]
    elif d < expected_dim:
        obs = np.concatenate([obs, np.zeros(expected_dim - d, dtype=np.float32)])
    return obs


def _normalize_model_path(p: str) -> str:
    return p[:-4] if p.lower().endswith(".zip") else p


def _resolve_default_model(train_mod: ModuleType) -> str:
    for attr in ("MODEL_SAVE_PATH", "MODEL_PATH", "BEST_MODEL_PATH"):
        p = getattr(train_mod, attr, None)
        if p and os.path.exists(f"{p}.zip"):
            return str(p)
    return str(getattr(train_mod, "MODEL_SAVE_PATH", "ppo_swarm_agent"))


# ===========================================================================
# SECTION 9 | SHARED EPISODE LOOP
# ===========================================================================

def _run_episode(
    env: Any,
    compute_action: Callable[[np.ndarray], np.ndarray],
    recorder: Optional[EpisodeRecorder],
    render_fn: Optional[Callable],
    render_every: int,
    show_final_only: bool,
) -> tuple[dict, list[dict]]:
    """
    Execute one episode.  Returns (metrics_dict, recorded_frames).

    compute_action(raw_obs) → action array  (closure handles LSTM state etc.)
    render_fn(step_num) is called every render_every steps if not None.
    """
    raw_obs, _ = env.reset()
    cum_rew    = 0.0
    speed_sum  = 0.0
    step       = 0
    done       = False
    info: dict = {}

    while not done:
        action = compute_action(raw_obs)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        cum_rew   += float(reward)
        speed_sum += abs(float(env._env.robots[0].v))
        step      += 1
        done       = bool(terminated or truncated)

        if recorder is not None:
            recorder.record(env, info)

        if render_fn is not None and not show_final_only:
            if step % max(render_every, 1) == 0:
                render_fn(step)

    if render_fn is not None and show_final_only:
        render_fn(step)

    active_found = float(info.get("active_found", info.get("found", 0)))
    active_total = float(info.get("active_total", info.get("total", 1)))
    end = (
        "SUCCESS"   if active_found >= active_total else
        "COLLISION" if info.get("collisions", 0) > 0 else
        "TIMEOUT"
    )
    result = {
        "reward":        cum_rew,
        "persons_found": active_found,
        "persons_total": active_total,
        "collisions":    float(info.get("collisions", 0)),
        "form_dev_mean": float(info.get("form_dev_mean", 0.0)),
        "form_dev_peak": float(info.get("form_dev_peak", info.get("form_dev_mean", 0.0))),
        "coverage_frac": float(info.get("coverage_frac", 0.0)),
        "steps":         float(step),
        "avg_speed":     float(speed_sum / max(step, 1)),
        "end":           end,
        "success":       active_found >= active_total and info.get("collisions", 0) == 0,
        "reward_components": dict(info.get("reward_components", {})),
        "completion_time":   float(step) * DT,
    }
    return result, list(recorder.frames) if recorder else []


def _print_episode(ep: int, n_eps: int, seed: int, result: dict,
                   train_mod: ModuleType) -> None:
    print(f"\n--- Episode {ep}/{n_eps}  (seed={seed}) ---")
    print(f"  Result    : {result['end']}")
    print(f"  Reward    : {result['reward']:.1f}")
    print(f"  Found     : {int(result['persons_found'])}/{int(result['persons_total'])}")
    print(f"  Collisions: {result['collisions']:.0f}")
    print(f"  Steps     : {int(result['steps'])}/{train_mod.MAX_EPISODE_STEPS}"
          f"  ({result['completion_time']:.1f} s)")
    print(f"  Coverage  : {result['coverage_frac']*100:.1f}%")
    rc = result.get("reward_components", {})
    if rc:
        print("  Reward breakdown:")
        for name, val in sorted(rc.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(val) > 0.01:
                print(f"    {name:>20s} : {val:>+9.1f}")


# ===========================================================================
# SECTION 10 | RL EVALUATION
# ===========================================================================

def evaluate_rl_model(
    train_mod: ModuleType,
    n_episodes: int,
    device: str,
    model_path: str,
    seed_base: int,
    n_persons: int,
    n_obstacles: int,
    render_every: int,
    show_final_only: bool,
    wait_for_input: bool,
    save_video: bool,
    all_episodes: bool,
    elev: float,
    azim: float,
    rotate: bool,
    random_seeds: bool = False,
    seed_range: int = 100000,
) -> dict[str, Any]:

    model_prefix = _normalize_model_path(model_path)
    model_file   = f"{model_prefix}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    print("\n" + "=" * 65)
    print(f"  RL EVALUATION | {n_episodes} episodes | {n_persons} persons, {n_obstacles} obstacles")
    print(f"  Model: {model_file}")
    print("=" * 65)

    model        = RecurrentPPO.load(model_prefix, device=device)
    expected_dim = int(model.observation_space.shape[0])
    vecnorm      = _load_vecnorm(train_mod, seed_base)
    warned_dim   = False

    results: list[dict]        = []
    all_frames: list[list[dict]] = []
    render_fig = render_ax = None

    for ep in range(n_episodes):
        seed = int(np.random.randint(0, seed_range)) if random_seeds else (seed_base + ep)
        env  = train_mod.SARGymnasiumWrapper(seed=seed)
        _configure_env(env, train_mod, n_persons, n_obstacles)

        # LSTM state reset per episode
        lstm_states = None
        ep_starts   = np.ones((1,), dtype=bool)

        def _rl_action(raw_obs: np.ndarray) -> np.ndarray:
            nonlocal lstm_states, ep_starts, warned_dim
            obs = _norm_obs(raw_obs, vecnorm, expected_dim)
            if not warned_dim and obs.shape[0] != expected_dim:
                print(f"  [!] Obs dim adapted: env={raw_obs.shape[0]} model={expected_dim}")
                warned_dim = True
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True,
            )
            ep_starts = np.zeros((1,), dtype=bool)
            return action

        recorder = EpisodeRecorder(n_persons) if (save_video or all_episodes) else None

        def _render(step_num: int) -> None:
            nonlocal render_fig, render_ax
            render_fig, render_ax = _render_2d_live(env, render_fig, render_ax)

        result, frames = _run_episode(
            env, _rl_action, recorder,
            render_fn=_render if (not save_video) else None,
            render_every=render_every,
            show_final_only=show_final_only,
        )
        results.append({**result, "episode": ep + 1, "seed": seed})
        all_frames.append(frames)

        _print_episode(ep + 1, n_episodes, seed, result, train_mod)

        if show_final_only and render_fig is not None:
            plt.pause(4.0 if not wait_for_input else 0.001)
        if wait_for_input:
            input("  [Enter for next episode] ")

        env.close()

        # Export video immediately if --all-episodes
        if all_episodes and frames:
            _export_video(frames, ep + 1, results[-1], train_mod,
                          n_persons, elev, azim, rotate, mode_tag="rl")

    if vecnorm is not None:
        vecnorm.close()
    if render_fig is not None:
        plt.close(render_fig)

    summary = _build_summary(results, model_file)
    _print_rl_summary(summary, train_mod)

    # Export best episode MP4
    if save_video and not all_episodes:
        best_i = _best_episode_index(results)
        if all_frames[best_i]:
            _export_video(all_frames[best_i], results[best_i]["episode"],
                          results[best_i], train_mod,
                          n_persons, elev, azim, rotate, mode_tag="rl")

    return summary


def _export_video(frames: list[dict], ep_num: int, result: dict,
                  train_mod: ModuleType, n_persons: int,
                  elev: float, azim: float, rotate: bool,
                  mode_tag: str = "rl") -> None:
    name = (
        f"{mode_tag}_ep{ep_num:03d}_seed{result['seed']}_"
        f"{result['end'].lower()}_"
        f"r{result['reward']:.0f}.mp4"
    )
    save_episode_mp4(frames, VIDEO_DIR / name, ep_num,
                     train_mod, n_persons, elev, azim, rotate)


def _best_episode_index(results: list[dict]) -> int:
    """Pick best episode: success first, then most found, then fewest steps."""
    def _key(r: dict) -> tuple:
        return (int(r["success"]), r["persons_found"], -r["steps"])
    return max(range(len(results)), key=lambda i: _key(results[i]))


def _build_summary(results: list[dict], model_file: str = "") -> dict:
    rewards     = [r["reward"]        for r in results]
    found_r     = [r["persons_found"] for r in results]
    total_r     = [r["persons_total"] for r in results]
    colls       = [r["collisions"]    for r in results]
    find_rates  = [
        float(r["persons_found"]) / max(float(r["persons_total"]), 1e-9)
        for r in results
    ]
    form_means  = [r["form_dev_mean"] for r in results]
    form_peaks  = [r.get("form_dev_peak", r["form_dev_mean"]) for r in results]
    covs        = [r["coverage_frac"] for r in results]
    speeds      = [r["avg_speed"]     for r in results]
    steps_l     = [r["steps"]         for r in results]
    successes   = [float(r["success"]) for r in results]
    complete_detection = [
        float(r["persons_found"] >= r["persons_total"])
        for r in results
    ]
    zero_collision = [float(r["collisions"] == 0) for r in results]
    return {
        "model_path":        model_file,
        "episodes_total":    len(results),
        "mean_reward":       float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        "mean_found":        float(np.mean(found_r)),
        "mean_total":        float(np.mean(total_r)),
        "find_rate_mean":    float(np.mean(find_rates)),
        "find_rate_std":     float(np.std(find_rates)),
        "mean_collisions":   float(np.mean(colls)),
        "std_collisions":    float(np.std(colls)),
        "form_dev_mean":     float(np.mean(form_means)),
        "form_dev_mean_std": float(np.std(form_means)),
        "form_dev_peak":     float(np.max(form_peaks) if form_peaks else 0.0),
        "form_dev_peak_std": float(np.std(form_peaks)),
        "mean_coverage":     float(np.mean(covs)),
        "mean_speed":        float(np.mean(speeds)),
        "mean_steps":        float(np.mean(steps_l)),
        "std_steps":         float(np.std(steps_l)),
        "complete_detection_rate": float(np.mean(complete_detection)),
        "zero_collision_episodes": int(np.sum(zero_collision)),
        "zero_collision_rate": float(np.mean(zero_collision)),
        "strict_success_rate": float(np.mean(successes)),
        "rows":              results,
    }


def _print_rl_summary(s: dict, train_mod: ModuleType) -> None:
    print("\n" + "=" * 65)
    print("  RL SUMMARY")
    print("=" * 65)
    if s["model_path"]:
        print(f"  Model            : {s['model_path']}")
    print(f"  Success rate     : {s['strict_success_rate']*100:.1f}%")
    print(f"  Mean reward      : {s['mean_reward']:.1f} ± {s['std_reward']:.1f}")
    print(f"  Mean found       : {s['mean_found']:.1f} / {s['mean_total']:.1f}")
    print(f"  Mean collisions  : {s['mean_collisions']:.2f} ± {s['std_collisions']:.2f}")
    print(f"  Mean coverage    : {s['mean_coverage']*100:.1f}%")
    print(f"  Mean speed       : {s['mean_speed']:.3f} m/s")
    print(f"  Mean steps/time  : {s['mean_steps']:.0f} / {s['mean_steps']*DT:.1f} s")
    print(f"  Find-rate (mean ± σ)           : {s['find_rate_mean']*100:.1f}% ± {s['find_rate_std']*100:.1f}%")
    print(f"  Formation error - mean (m)     : {s['form_dev_mean']:.3f} ± {s['form_dev_mean_std']:.3f}")
    print(f"  Formation error - peak (m)     : {s['form_dev_peak']:.3f} ± {s['form_dev_peak_std']:.3f}")
    print(f"  Episode duration - steps (mean): {s['mean_steps']:.1f} ± {s['std_steps']:.1f}")
    print(f"  Complete detection rate        : {s['complete_detection_rate']*100:.1f}%")
    print(f"  Episodes with 0 collisions     : {s['zero_collision_episodes']}/{s['episodes_total']} ({s['zero_collision_rate']*100:.1f}%)")
    print("=" * 65)


# ===========================================================================
# SECTION 11 | CLASSICAL BASELINE
# ===========================================================================

def evaluate_classical_baseline(
    train_mod: ModuleType,
    n_episodes: int,
    seed_base: int,
    n_persons: int,
    n_obstacles: int,
    render_every: int,
    show_final_only: bool,
    wait_for_input: bool,
    save_video: bool,
    all_episodes: bool,
    elev: float,
    azim: float,
    rotate: bool,
    random_seeds: bool = False,
    seed_range: int = 100000,
) -> dict[str, Any]:

    print("\n" + "=" * 65)
    print(f"  CLASSICAL BASELINE | {n_episodes} episodes | {n_persons} persons, {n_obstacles} obstacles")
    print("=" * 65)

    results: list[dict] = []
    all_frames: list[list[dict]] = []
    render_fig = render_ax = None

    for ep in range(n_episodes):
        seed = int(np.random.randint(0, seed_range)) if random_seeds else (seed_base + ep)
        env  = train_mod.SARGymnasiumWrapper(seed=seed)
        _configure_env(env, train_mod, n_persons, n_obstacles)

        leader_ctrl = LeaderCtrl(env._env.waypoints)

        def _classical_action(raw_obs: np.ndarray) -> np.ndarray:
            leader    = env._env.robots[0]
            known_obs = env._env.shared_obs.all()
            v_l, w_l  = leader_ctrl(leader, known_obs)
            a0 = float(np.clip(2.0 * v_l / max(train_mod.V_MAX, 1e-9) - 1.0, -1.0, 1.0))
            a1 = float(np.clip(w_l / max(train_mod.OMEGA_MAX, 1e-9), -1.0, 1.0))
            return np.array([a0, a1], dtype=np.float32)

        def _render(step_num: int) -> None:
            nonlocal render_fig, render_ax
            render_fig, render_ax = _render_2d_live(env, render_fig, render_ax)

        recorder = EpisodeRecorder(n_persons) if (save_video or all_episodes) else None

        result, frames = _run_episode(
            env, _classical_action, recorder=recorder,
            render_fn=_render if (not save_video) else None,
            render_every=render_every,
            show_final_only=show_final_only,
        )
        results.append({**result, "episode": ep + 1, "seed": seed})
        all_frames.append(frames)

        _print_episode(ep + 1, n_episodes, seed, result, train_mod)

        if show_final_only and render_fig is not None:
            plt.pause(4.0 if not wait_for_input else 0.001)
        if wait_for_input:
            input("  [Enter for next episode] ")
        env.close()

        if all_episodes and frames:
            _export_video(frames, ep + 1, results[-1], train_mod,
                          n_persons, elev, azim, rotate, mode_tag="classical")

    if render_fig is not None:
        plt.close(render_fig)

    summary = _build_summary(results)
    print("\n" + "=" * 65)
    print("  CLASSICAL SUMMARY")
    print("=" * 65)
    print(f"  Success rate     : {summary['strict_success_rate']*100:.1f}%")
    print(f"  Mean reward      : {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
    print(f"  Mean found       : {summary['mean_found']:.1f} / {summary['mean_total']:.1f}")
    print(f"  Mean collisions  : {summary['mean_collisions']:.2f} ± {summary['std_collisions']:.2f}")
    print(f"  Mean steps/time  : {summary['mean_steps']:.0f} / {summary['mean_steps']*DT:.1f} s")
    print(f"  Find-rate (mean ± σ)           : {summary['find_rate_mean']*100:.1f}% ± {summary['find_rate_std']*100:.1f}%")
    print(f"  Formation error - mean (m)     : {summary['form_dev_mean']:.3f} ± {summary['form_dev_mean_std']:.3f}")
    print(f"  Formation error - peak (m)     : {summary['form_dev_peak']:.3f} ± {summary['form_dev_peak_std']:.3f}")
    print(f"  Episode duration - steps (mean): {summary['mean_steps']:.1f} ± {summary['std_steps']:.1f}")
    print(f"  Complete detection rate        : {summary['complete_detection_rate']*100:.1f}%")
    print(f"  Episodes with 0 collisions     : {summary['zero_collision_episodes']}/{summary['episodes_total']} ({summary['zero_collision_rate']*100:.1f}%)")
    print("=" * 65)

    if save_video and not all_episodes:
        best_i = _best_episode_index(results)
        if all_frames[best_i]:
            _export_video(all_frames[best_i], results[best_i]["episode"],
                          results[best_i], train_mod,
                          n_persons, elev, azim, rotate, mode_tag="classical")

    return summary


# ===========================================================================
# SECTION 11B | SEED GENERATOR
# ===========================================================================

def _generate_seeds(seed_base: int, n_episodes: int, random_seeds: bool, seed_range: int) -> list[int]:
    """Generate episode seeds either sequentially or randomly."""
    if random_seeds:
        return [int(np.random.randint(0, seed_range)) for _ in range(n_episodes)]
    else:
        return [seed_base + ep for ep in range(n_episodes)]


# ===========================================================================
# SECTION 12 | TOP-LEVEL RUNNER
# ===========================================================================

def run_evaluation(
    train_mod: ModuleType,
    n_episodes: int,
    device: str,
    wait_for_input: bool,
    model_paths: list[str],
    mode: str,
    seed_base: int,
    random_seeds: bool,
    seed_range: int,
    override_persons: Optional[int],
    override_obstacles: Optional[int],
    render_every: int,
    show_final_only: bool,
    save_video: bool,
    all_episodes: bool,
    elev: float,
    azim: float,
    rotate: bool,
) -> None:

    n_persons, n_obstacles = _get_active_counts(
        train_mod, override_persons, override_obstacles
    )

    rl_summaries: list[dict] = []
    if mode in ("rl", "both"):
        models = model_paths or [_resolve_default_model(train_mod)]
        for mp in models:
            rl_summaries.append(evaluate_rl_model(
                train_mod=train_mod,
                n_episodes=n_episodes,
                device=device,
                model_path=mp,
                seed_base=seed_base,
                n_persons=n_persons,
                n_obstacles=n_obstacles,
                render_every=render_every,
                show_final_only=show_final_only,
                wait_for_input=wait_for_input,
                save_video=save_video,
                all_episodes=all_episodes,
                elev=elev,
                azim=azim,
                random_seeds=random_seeds,
                seed_range=seed_range,
                rotate=rotate,
            ))

    classical: Optional[dict] = None
    if mode in ("classical", "both"):
        classical = evaluate_classical_baseline(
            train_mod=train_mod,
            n_episodes=n_episodes,
            seed_base=seed_base,
            n_persons=n_persons,
            n_obstacles=n_obstacles,
            render_every=render_every,
            show_final_only=show_final_only,
            wait_for_input=wait_for_input,
            save_video=save_video,
            all_episodes=all_episodes,
            elev=elev,
            azim=azim,
            rotate=rotate,
            random_seeds=random_seeds,
            seed_range=seed_range,
        )

    # Multi-model comparison table
    if len(rl_summaries) > 1:
        print("\n" + "=" * 90)
        print("  RL MODEL COMPARISON")
        print("=" * 90)
        print(f"  {'Model':<40} {'Reward':>10} {'Found':>12} {'Coll':>8} {'Time(s)':>10} {'Success':>8}")
        print("  " + "-" * 88)
        for s in rl_summaries:
            print(f"  {s['model_path']:<40} {s['mean_reward']:>10.1f}"
                  f" {s['mean_found']:>5.1f}/{s['mean_total']:<5.1f}"
                  f" {s['mean_collisions']:>8.2f}"
                  f" {s['mean_steps']*DT:>10.1f}"
                  f" {s['strict_success_rate']*100:>7.1f}%")
        print("=" * 90)

    # RL vs Classical comparison + plots
    if classical is not None and rl_summaries:
        for s in rl_summaries:
            print("\n" + "=" * 65)
            print(f"  RL VS CLASSICAL  |  model={s['model_path']}")
            print("=" * 65)
            print(f"  Reward    : RL {s['mean_reward']:.1f}±{s['std_reward']:.1f}"
                  f"  |  Classical {classical['mean_reward']:.1f}±{classical['std_reward']:.1f}")
            print(f"  Found     : RL {s['mean_found']:.1f}/{s['mean_total']:.1f}"
                  f"  |  Classical {classical['mean_found']:.1f}/{classical['mean_total']:.1f}")
            print(f"  Collisions: RL {s['mean_collisions']:.2f}"
                  f"  |  Classical {classical['mean_collisions']:.2f}")
            print(f"  Time      : RL {s['mean_steps']*DT:.1f}s"
                  f"  |  Classical {classical['mean_steps']*DT:.1f}s")
            print(f"  Success   : RL {s['strict_success_rate']*100:.1f}%"
                  f"  |  Classical {classical['strict_success_rate']*100:.1f}%")
            print("=" * 65)

            # Build uniform plot rows
            rl_plot   = [{"reward": r["reward"], "form_dev_mean": r["form_dev_mean"],
                           "completion_time": r["completion_time"],
                           "persons_found": r["persons_found"],
                           "persons_total": r["persons_total"]}
                          for r in s["rows"]]
            cl_plot   = [{"reward": r["reward"], "form_dev_mean": r["form_dev_mean"],
                           "completion_time": r["completion_time"],
                           "persons_found": r["persons_found"],
                           "persons_total": r["persons_total"]}
                          for r in classical["rows"]]
            if len(rl_plot) == len(cl_plot) and rl_plot:
                lp, ap = _save_comparison_plots(rl_plot, cl_plot, seed_base, n_episodes)
                print(f"  Saved plots: {lp}  |  {ap}")
                if os.name == "nt":
                    try:
                        os.startfile(str(lp))
                        os.startfile(str(ap))
                    except Exception:
                        pass


# ===========================================================================
# SECTION 13 | CLI + ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SAR Swarm Evaluator — RL model, classical baseline, and 3D MP4 export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core
    parser.add_argument("--mode", choices=["rl", "classical", "both"],
                        default="both", help="Evaluation mode")
    parser.add_argument("--train-file", default="train_rl.py",
                        help="Training script to import settings from")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Evaluation episodes per model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto", help="Torch device")
    parser.add_argument("--model-path", action="append", dest="model_paths",
                        default=None, metavar="PATH",
                        help="RL model path (repeat to compare multiple)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="First episode seed (default: training SEED)")
    parser.add_argument("--random-seeds", action="store_true",
                        help="Use random seeds instead of sequential (seed-base to seed-base+episodes)")
    parser.add_argument("--seed-range", type=int, default=100000,
                        help="Max seed value when using --random-seeds (default: 100000)")
    parser.add_argument("--eval-persons", type=int, default=None,
                        help="Override active persons count")
    parser.add_argument("--eval-obstacles", type=int, default=None,
                        help="Override active obstacles count")
    # Rendering
    parser.add_argument("--render-every", type=int, default=30,
                        help="2D live render interval (steps)")
    parser.add_argument("--show-final-only", action="store_true",
                        help="Only show 2D render at episode end")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip Enter prompts between episodes")
    # 3D video
    parser.add_argument("--no-video", action="store_true",
                        help="Skip MP4 export entirely")
    parser.add_argument("--all-episodes", action="store_true",
                        help="Export MP4 for every episode (default: best only)")
    parser.add_argument("--elev", type=float, default=35.0,
                        help="3D camera elevation (degrees)")
    parser.add_argument("--azim", type=float, default=-55.0,
                        help="3D camera azimuth (degrees)")
    parser.add_argument("--rotate", action="store_true",
                        help="Slowly rotate camera during 3D animation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mod = load_train_module(args.train_file)
    seed_base = (train_mod.SEED if args.seed_base is None else int(args.seed_base))

    run_evaluation(
        train_mod          = train_mod,
        n_episodes         = args.episodes,
        device             = args.device,
        wait_for_input     = not args.no_prompt,
        model_paths        = args.model_paths or [],
        mode               = args.mode,
        seed_base          = seed_base,
        random_seeds       = args.random_seeds,
        seed_range         = args.seed_range,
        override_persons   = args.eval_persons,
        override_obstacles = args.eval_obstacles,
        render_every       = max(args.render_every, 1),
        show_final_only    = args.show_final_only,
        save_video         = not args.no_video,
        all_episodes       = args.all_episodes,
        elev               = args.elev,
        azim               = args.azim,
        rotate             = args.rotate,
    )


if __name__ == "__main__":
    main()
