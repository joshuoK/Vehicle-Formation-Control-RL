"""
eval_v10.py — Evaluation + 3D Mission Visualiser for v10

Runs N evaluation episodes with the v10 model, prints per-episode metrics
and a summary, then saves the best episode as an MP4 3D animation showing
robots navigating the map, discovering persons, and building coverage.

Usage:
  python eval_v10.py                             # 5 eps, auto-find model
  python eval_v10.py --episodes 20               # more episodes
  python eval_v10.py --model checkpoints_v10/best_model
  python eval_v10.py --model ppo_swarm_agent_v10
  python eval_v10.py --no-video                  # skip MP4 export
  python eval_v10.py --all-episodes              # export MP4 for EVERY episode
  python eval_v10.py --device cpu                # force CPU inference
  python eval_v10.py --seed 42                   # fixed episode seed

Requires FFmpeg for MP4 output.  Install:
  Windows : https://ffmpeg.org/download.html  (add ffmpeg/bin to PATH)
  or      : conda install -c conda-forge ffmpeg
  or      : pip install imageio-ffmpeg  (alternative writer)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reliable off-screen rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# ── SB3 imports ────────────────────────────────────────────────────────────
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import train_rl_v10 as T   # noqa: E402  (import after path setup)
from sar_environment import MAP_W, MAP_H, DT, FOV_ANG, FOV_RANGE


# ===========================================================================
# CONSTANTS
# ===========================================================================

ROBOT_COLORS   = ["#4fc3f7", "#ef5350", "#ffa726"]   # leader, f1, f2
ROBOT_LABELS   = ["Leader", "Follower-1", "Follower-2"]
PERSON_COLOR_UNDET = "#ffee58"   # yellow — not yet found
PERSON_COLOR_DET   = "#00e676"   # green  — found
OBS_COLOR_KNOWN    = "#4a4a8c"
OBS_COLOR_HIDDEN   = "#2a2a4c"
COVER_COLOR        = "#00e676"
BG_COLOR           = "#0a0a14"

FRAME_SKIP = 5          # subsample: render every Nth step (lower = smoother but larger MP4)
VIDEO_FPS  = 24         # frames per second in output MP4
ROBOT_Z    = 0.45       # robot marker height above floor [m]
TRAIL_ALPHA = 0.45
COVER_ALPHA = 0.30
OBSTACLE_HEIGHT_SCALE = 2.0   # visual height = radius * this
HALF_FOV = float(0.5 * FOV_ANG)

OUTPUT_DIR = Path("eval_videos_v10")


# ===========================================================================
# EPISODE RECORDER
# ===========================================================================
class EpisodeRecorder:
    """Captures the state of the environment every FRAME_SKIP steps."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self._step = 0

    def record(self, env: T.SARGymnasiumWrapper, extra: dict) -> None:
        """Call this every env.step(); records if on the right frame interval."""
        self._step += 1
        if self._step % FRAME_SKIP != 0:
            return

        robots = [
            (float(r.x), float(r.y), float(r.theta))
            for r in env._env.robots
        ]
        # Snapshot the trail (copy list of tuples)
        trails = [list(r.trail) for r in env._env.robots]

        persons = [
            (float(p.x), float(p.y), bool(p.detected))
            for p in env._env.persons[:T.ACTIVE_PERSONS]
        ]
        known_keys = set(env._env.shared_obs._cells.keys())
        obstacles = []
        for ox, oy, rad in env._env.true_obstacles[:T.ACTIVE_OBSTACLES]:
            k = env._env.shared_obs._key(ox, oy)
            obstacles.append((float(ox), float(oy), float(rad), k in known_keys))

        self.frames.append({
            "step":          self._step,
            "robots":        robots,
            "trails":        trails,
            "persons":       persons,
            "obstacles":     obstacles,
            "coverage_grid": env._coverage_grid.copy(),  # bool array (400,)
            "found":         extra.get("active_found", 0),
            "coverage_frac": extra.get("coverage_frac", 0.0),
        })

    def reset(self) -> None:
        self.frames.clear()
        self._step = 0


# ===========================================================================
# 3D FRAME RENDERER
# ===========================================================================

def _draw_coverage(ax: Axes3D, coverage_grid: np.ndarray) -> None:
    """Draw covered cells as thin green pillars on the floor."""
    covered_idx = np.flatnonzero(coverage_grid)
    if len(covered_idx) == 0:
        return
    ci = covered_idx // T.GRID_N
    cj = covered_idx %  T.GRID_N
    xs = ci * T.CELL_SIZE_X
    ys = cj * T.CELL_SIZE_Y
    dz = np.full(len(xs), 0.12)
    ax.bar3d(
        xs, ys, np.zeros(len(xs)),
        T.CELL_SIZE_X, T.CELL_SIZE_Y, dz,
        color=COVER_COLOR, alpha=COVER_ALPHA,
        shade=False, zsort="min",
    )


def _draw_obstacle_cylinder(ax: Axes3D, ox: float, oy: float,
                             rad: float, is_known: bool) -> None:
    """Draw an obstacle as a 3D cylinder."""
    height = rad * OBSTACLE_HEIGHT_SCALE
    color  = OBS_COLOR_KNOWN if is_known else OBS_COLOR_HIDDEN
    alpha  = 0.85 if is_known else 0.35
    theta  = np.linspace(0, 2 * np.pi, 20)
    z_cyl  = np.linspace(0.0, height, 4)
    Theta, Z = np.meshgrid(theta, z_cyl)
    X = ox + rad * np.cos(Theta)
    Y = oy + rad * np.sin(Theta)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=False)
    # Top cap
    cap_r = np.linspace(0, rad, 5)
    Cap_T, Cap_R = np.meshgrid(theta, cap_r)
    ax.plot_surface(
        ox + Cap_R * np.cos(Cap_T),
        oy + Cap_R * np.sin(Cap_T),
        np.full_like(Cap_T, height),
        color=color, alpha=alpha, linewidth=0, antialiased=False,
    )


def _draw_fov_arc(ax: Axes3D, rx: float, ry: float, rtheta: float,
                  sensor_off: float, color: str) -> None:
    """Project each robot's FOV cone as an arc on the floor."""
    hdg = rtheta + sensor_off
    arc_theta = np.linspace(hdg - HALF_FOV, hdg + HALF_FOV, 20)
    arc_x = [rx] + (rx + FOV_RANGE * np.cos(arc_theta)).tolist() + [rx]
    arc_y = [ry] + (ry + FOV_RANGE * np.sin(arc_theta)).tolist() + [ry]
    arc_z = [0.02] * len(arc_x)
    ax.plot3D(arc_x, arc_y, arc_z, color=color, alpha=0.18, lw=0.8)


def _draw_robot(ax: Axes3D, rx: float, ry: float, rtheta: float,
                color: str, label: str) -> None:
    """Draw a robot body + heading arrow at ROBOT_Z height."""
    # Body sphere
    ax.scatter3D([rx], [ry], [ROBOT_Z], s=260, c=color,
                 marker="o", edgecolors="white", linewidths=0.6,
                 depthshade=False, zorder=10)
    # Heading arrow (short line pointing forward)
    arr_len = 0.9
    ax.plot3D(
        [rx, rx + arr_len * np.cos(rtheta)],
        [ry, ry + arr_len * np.sin(rtheta)],
        [ROBOT_Z, ROBOT_Z],
        color=color, lw=2.0, zorder=11,
    )
    # Vertical drop line (connects robot to floor)
    ax.plot3D([rx, rx], [ry, ry], [0, ROBOT_Z],
              color=color, alpha=0.25, lw=0.8, ls="--")


def render_frame_3d(ax: Axes3D, frame: dict, episode_num: int) -> None:
    """Clear and redraw the 3D scene for one recorded frame."""
    ax.cla()

    # ── Axes setup ────────────────────────────────────────────────────────
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(0, MAP_H)
    ax.set_zlim(0, 5.0)
    ax.set_facecolor(BG_COLOR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1e1e35")
    ax.yaxis.pane.set_edgecolor("#1e1e35")
    ax.zaxis.pane.set_edgecolor("#1e1e35")
    ax.set_xlabel("X [m]", color="#666", fontsize=7, labelpad=2)
    ax.set_ylabel("Y [m]", color="#666", fontsize=7, labelpad=2)
    ax.set_zlabel("",        color="#666", fontsize=7)
    ax.tick_params(colors="#555", labelsize=6)

    # Floor outline
    bx = [0, MAP_W, MAP_W, 0, 0]
    by = [0, 0, MAP_H, MAP_H, 0]
    ax.plot3D(bx, by, [0]*5, color="#334", lw=1.0)

    # Floor grid (5m spacing)
    for v in np.arange(0, MAP_W + 1, 5):
        ax.plot3D([v, v], [0, MAP_H], [0, 0], color="#1a1a2e", lw=0.4)
    for h in np.arange(0, MAP_H + 1, 5):
        ax.plot3D([0, MAP_W], [h, h], [0, 0], color="#1a1a2e", lw=0.4)

    # ── Coverage ──────────────────────────────────────────────────────────
    _draw_coverage(ax, frame["coverage_grid"])

    # ── Obstacles ─────────────────────────────────────────────────────────
    for ox, oy, rad, is_known in frame["obstacles"]:
        _draw_obstacle_cylinder(ax, ox, oy, rad, is_known)

    # ── Persons ───────────────────────────────────────────────────────────
    from sar_environment import SENSOR_OFF
    for px, py, detected in frame["persons"]:
        if detected:
            ax.scatter3D([px], [py], [0.15], s=260, c=PERSON_COLOR_DET,
                         marker="*", depthshade=False, zorder=9)
            # Glow ring on the floor
            ring_t = np.linspace(0, 2 * np.pi, 30)
            ax.plot3D(
                px + 0.6 * np.cos(ring_t), py + 0.6 * np.sin(ring_t),
                [0.02] * 30, color=PERSON_COLOR_DET, alpha=0.5, lw=0.8,
            )
        else:
            ax.scatter3D([px], [py], [0.1], s=180, c=PERSON_COLOR_UNDET,
                         marker="P", depthshade=False, zorder=8)

    # ── Robot FOV arcs, trails, bodies ────────────────────────────────────
    sensor_offsets = SENSOR_OFF  # [0, +120°, -120°]
    for i, ((rx, ry, rtheta), color, trail) in enumerate(
        zip(frame["robots"], ROBOT_COLORS, frame["trails"])
    ):
        # FOV arc on the floor
        _draw_fov_arc(ax, rx, ry, rtheta, float(sensor_offsets[i]), color)

        # Trail
        if len(trail) > 2:
            trail_arr = np.array(trail[-400:])  # cap trail length for speed
            ax.plot3D(trail_arr[:, 0], trail_arr[:, 1],
                      np.full(len(trail_arr), 0.05),
                      color=color, alpha=TRAIL_ALPHA, lw=0.8)

        # Robot body
        _draw_robot(ax, rx, ry, rtheta, color, ROBOT_LABELS[i])

    # ── HUD title ─────────────────────────────────────────────────────────
    t_sec      = frame["step"] * DT
    found      = int(frame["found"])
    cov_pct    = frame["coverage_frac"] * 100.0
    ax.set_title(
        f"Episode {episode_num}  |  t = {t_sec:.1f} s  "
        f"found = {found}/{T.ACTIVE_PERSONS}  "
        f"coverage = {cov_pct:.0f}%",
        color="white", fontsize=9, pad=8,
    )

    # ── Legend (drawn once per frame — small overhead) ────────────────────
    legend_items = (
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
            plt.Rectangle((0, 0), 1, 1, fc=COVER_COLOR, alpha=0.5,
                           label="Coverage"),
        ]
    )
    ax.legend(
        handles=legend_items, loc="upper left",
        fontsize=6, facecolor="#0e0e20", edgecolor="#4fc3f7",
        labelcolor="white", framealpha=0.7,
    )


# ===========================================================================
# MP4 EXPORT
# ===========================================================================

def save_episode_mp4(
    frames: list[dict],
    output_path: Path,
    episode_num: int,
    elev: float = 35.0,
    azim: float = -55.0,
    rotate: bool = False,
) -> None:
    """
    Render all recorded frames into a 3D animation and write to MP4.

    Args:
        frames:       List of frame dicts from EpisodeRecorder.
        output_path:  Destination .mp4 file path.
        episode_num:  Episode number shown in the title HUD.
        elev:         Camera elevation angle (degrees above XY plane).
        azim:         Camera azimuth angle (degrees).
        rotate:       If True, slowly rotate the camera azimuth during playback.
    """
    if not frames:
        print("[!] No frames recorded — skipping MP4 export.")
        return

    print(f"\n  Rendering {len(frames)} frames → {output_path}")
    print(f"  Camera: elev={elev}°  azim={azim}°  rotate={rotate}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9), facecolor=BG_COLOR)
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    pbar_width = 40

    def _update(frame_idx: int) -> None:
        frame = frames[frame_idx]
        render_frame_3d(ax, frame, episode_num)
        if rotate:
            current_azim = azim + frame_idx * (180.0 / max(len(frames), 1))
            ax.view_init(elev=elev, azim=current_azim)
        # Progress bar
        pct = (frame_idx + 1) / len(frames)
        filled = int(pbar_width * pct)
        bar = "#" * filled + "-" * (pbar_width - filled)
        sys.stdout.write(f"\r  [{bar}] {pct*100:.0f}%  frame {frame_idx+1}/{len(frames)}")
        sys.stdout.flush()

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(frames),
        interval=1000 // VIDEO_FPS,
        repeat=False,
        blit=False,
    )

    # Try FFMpeg writer first, fall back to Pillow (GIF)
    try:
        writer = animation.FFMpegWriter(
            fps=VIDEO_FPS,
            metadata={"title": "SAR v10 Eval", "artist": "train_rl_v10"},
            bitrate=2500,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],  # broad player compatibility
        )
        anim.save(str(output_path), writer=writer, dpi=120)
        print(f"\n  MP4 saved → {output_path}")
    except (FileNotFoundError, RuntimeError) as ffmpeg_err:
        print(f"\n  [!] FFMpeg not found ({ffmpeg_err}).")
        gif_path = output_path.with_suffix(".gif")
        print(f"  Falling back to GIF → {gif_path}")
        try:
            writer_gif = animation.PillowWriter(fps=VIDEO_FPS)
            anim.save(str(gif_path), writer=writer_gif, dpi=90)
            print(f"  GIF saved → {gif_path}")
            print("  Install FFmpeg to get MP4: https://ffmpeg.org/download.html")
        except Exception as gif_err:
            print(f"  [!] GIF export also failed: {gif_err}")
            print("  Install Pillow: pip install Pillow")

    plt.close(fig)


# ===========================================================================
# EVALUATION LOOP
# ===========================================================================

def _normalize_model_path(path: str) -> str:
    return path[:-4] if path.lower().endswith(".zip") else path


def _load_vecnorm(model_env: T.SARGymnasiumWrapper, seed: int) -> Optional[VecNormalize]:
    """Try to load VecNormalize stats from the v10 checkpoint directory."""
    vecnorm_path = Path(T.VECNORM_PATH)
    if not vecnorm_path.exists():
        print(f"  [!] VecNormalize not found at '{vecnorm_path}' — using raw obs.")
        return None
    base = DummyVecEnv([lambda: model_env])
    vn = VecNormalize.load(str(vecnorm_path), base)
    vn.training   = False
    vn.norm_reward = False
    return vn


def _normalize_obs(obs: np.ndarray, vn: Optional[VecNormalize],
                   expected_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if vn is not None:
        obs = np.asarray(vn.normalize_obs(obs[None, :])[0], dtype=np.float32)
    # Adapt dimension if obs space changed between checkpoint and current env
    if obs.shape[0] != expected_dim:
        if obs.shape[0] > expected_dim:
            obs = obs[:expected_dim]
        else:
            obs = np.concatenate([obs, np.zeros(expected_dim - obs.shape[0], dtype=np.float32)])
    return obs


def run_eval(
    model_path: str,
    n_episodes: int,
    device: str,
    seed_base: int,
    save_video: bool,
    all_episodes: bool,
    elev: float,
    azim: float,
    rotate: bool,
) -> None:
    model_prefix = _normalize_model_path(model_path)
    model_file   = f"{model_prefix}.zip"

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Model not found: {model_file}\n"
            f"Train first: python train_rl_v10.py --mode train"
        )

    print("=" * 65)
    print("  SAR v10 EVALUATION")
    print("=" * 65)
    print(f"  Model   : {model_file}")
    print(f"  Mission : {T.ACTIVE_PERSONS} persons, {T.ACTIVE_OBSTACLES} obstacles")
    print(f"  Episodes: {n_episodes}  seeds {seed_base} … {seed_base + n_episodes - 1}")
    print(f"  Device  : {device}")
    print(f"  Video   : {'yes (all)' if all_episodes else 'yes (best)' if save_video else 'no'}")
    print("=" * 65)

    model = RecurrentPPO.load(model_prefix, device=device)
    expected_dim = int(model.observation_space.shape[0])

    # VecNormalize needs a throwaway env — close it immediately after loading
    _tmp_env = T.SARGymnasiumWrapper(seed=seed_base)
    vecnorm  = _load_vecnorm(_tmp_env, seed_base)
    # Don't close _tmp_env yet if it's inside vecnorm; vecnorm owns it
    # If no vecnorm, close manually
    if vecnorm is None:
        _tmp_env.close()

    results: list[dict] = []
    all_frames: list[list[dict]] = []
    recorder   = EpisodeRecorder()

    for ep in range(n_episodes):
        seed = seed_base + ep
        print(f"\n--- Episode {ep + 1}/{n_episodes}  (seed={seed}) ---")

        env = T.SARGymnasiumWrapper(seed=seed)
        env.set_collision_grace_active(False)  # eval: full difficulty
        raw_obs, _ = env.reset()

        obs            = _normalize_obs(raw_obs, vecnorm, expected_dim)
        lstm_states    = None
        ep_starts      = np.ones((1,), dtype=bool)
        cumulative_rew = 0.0
        done           = False
        step_num       = 0
        info: dict     = {}
        recorder.reset()

        t0 = time.time()
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=ep_starts, deterministic=True,
            )
            raw_obs, reward, terminated, truncated, info = env.step(action)
            cumulative_rew += float(reward)
            done = bool(terminated or truncated)
            step_num += 1
            ep_starts = np.zeros((1,), dtype=bool)
            if not done:
                obs = _normalize_obs(raw_obs, vecnorm, expected_dim)

            # Record state for animation
            if save_video or all_episodes:
                recorder.record(env, info)

        wall_time = time.time() - t0

        active_found = float(info.get("active_found", info.get("found", 0)))
        active_total = float(info.get("active_total", info.get("total", T.ACTIVE_PERSONS)))
        success = (active_found >= active_total) and (info.get("collisions", 0) == 0)

        if active_found >= active_total:
            end = "SUCCESS"
        elif info.get("collisions", 0) > 0:
            end = "COLLISION"
        else:
            end = "TIMEOUT"

        results.append({
            "episode":       ep + 1,
            "seed":          seed,
            "reward":        cumulative_rew,
            "persons_found": active_found,
            "persons_total": active_total,
            "collisions":    float(info.get("collisions", 0)),
            "form_dev":      float(info.get("form_dev_mean", 0.0)),
            "coverage_frac": float(info.get("coverage_frac", 0.0)),
            "steps":         float(step_num),
            "success":       success,
            "end":           end,
            "wall_time":     wall_time,
        })
        all_frames.append(list(recorder.frames))

        print(f"  Result    : {end}")
        print(f"  Reward    : {cumulative_rew:.1f}")
        print(f"  Found     : {int(active_found)}/{int(active_total)}")
        print(f"  Collisions: {info.get('collisions', 0)}")
        print(f"  Steps     : {step_num}/{T.MAX_EPISODE_STEPS}  ({step_num*DT:.1f} s)")
        print(f"  Coverage  : {info.get('coverage_frac', 0.0)*100:.1f}%")
        rc = info.get("reward_components", {})
        if rc:
            print("  Reward breakdown:")
            for name, val in sorted(rc.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(val) > 0.01:
                    print(f"    {name:>20s} : {val:>+9.1f}")

        env.close()

        # Export this episode immediately if --all-episodes
        if all_episodes and all_frames[-1]:
            mp4_path = OUTPUT_DIR / f"episode_{ep+1:03d}_seed{seed}.mp4"
            save_episode_mp4(all_frames[-1], mp4_path, ep + 1, elev, azim, rotate)

    if vecnorm is not None:
        vecnorm.close()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    rewards     = [r["reward"]        for r in results]
    found_rates = [r["persons_found"] / max(r["persons_total"], 1) for r in results]
    successes   = [r["success"]       for r in results]
    steps_list  = [r["steps"]         for r in results]
    collisions  = [r["collisions"]    for r in results]
    coverages   = [r["coverage_frac"] for r in results]

    print(f"  Episodes        : {n_episodes}")
    print(f"  Success rate    : {np.mean(successes)*100:.1f}%  ({int(np.sum(successes))}/{n_episodes})")
    print(f"  Mean reward     : {np.mean(rewards):.1f}  ±  {np.std(rewards):.1f}")
    print(f"  Mean find rate  : {np.mean(found_rates)*100:.1f}%")
    print(f"  Mean collisions : {np.mean(collisions):.2f}")
    print(f"  Mean coverage   : {np.mean(coverages)*100:.1f}%")
    print(f"  Mean steps      : {np.mean(steps_list):.0f}  ({np.mean(steps_list)*DT:.1f} s)")
    print(f"  Best reward     : {np.max(rewards):.1f}  (ep {np.argmax(rewards)+1})")

    # Success breakdown
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(
            f"  {mark} ep{r['episode']:>2d} "
            f"seed={r['seed']} "
            f"  {r['end']:>10s} "
            f"  reward={r['reward']:>7.1f} "
            f"  found={int(r['persons_found'])}/{int(r['persons_total'])} "
            f"  cov={r['coverage_frac']*100:.0f}%"
        )
    print("=" * 65)

    # ── Best-episode MP4 ───────────────────────────────────────────────────
    if save_video and not all_episodes:
        # Pick the best episode: prefer success + most found + fastest
        def _score(r: dict) -> tuple:
            return (
                int(r["success"]),          # success first
                r["persons_found"],          # most persons found
                -r["steps"],                 # fewest steps (negative = faster is better)
            )

        best_idx  = max(range(len(results)), key=lambda i: _score(results[i]))
        best_res  = results[best_idx]
        best_frames = all_frames[best_idx]

        print(f"\n  Best episode: {best_idx+1} ({best_res['end']}, "
              f"reward={best_res['reward']:.1f}, "
              f"found={int(best_res['persons_found'])}/{int(best_res['persons_total'])})")

        mp4_name = (
            f"best_ep{best_res['episode']:03d}_"
            f"seed{best_res['seed']}_"
            f"{best_res['end'].lower()}.mp4"
        )
        mp4_path = OUTPUT_DIR / mp4_name
        save_episode_mp4(best_frames, mp4_path, best_res["episode"], elev, azim, rotate)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _default_model_path() -> str:
    """Find the best available v10 model automatically."""
    candidates = [
        T.BEST_MODEL_PATH,          # checkpoints_v10/best_model
        T.MODEL_SAVE_PATH,          # ppo_swarm_agent_v10
    ]
    for p in candidates:
        if os.path.exists(f"{p}.zip"):
            return p
    # If nothing found, return the primary path (will raise FileNotFoundError later)
    return T.MODEL_SAVE_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v10 SAR model and export 3D MP4 visualisation."
    )
    parser.add_argument(
        "--model", default=None,
        help=("Path to model (with or without .zip). "
              "Auto-detected if omitted."),
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Torch device (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=T.SEED,
        help=f"Starting seed (default: {T.SEED})",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip MP4 export (eval metrics only)",
    )
    parser.add_argument(
        "--all-episodes", action="store_true",
        help="Export MP4 for every episode (default: best only)",
    )
    parser.add_argument(
        "--elev", type=float, default=35.0,
        help="Camera elevation angle in degrees (default: 35)",
    )
    parser.add_argument(
        "--azim", type=float, default=-55.0,
        help="Camera azimuth angle in degrees (default: -55)",
    )
    parser.add_argument(
        "--rotate", action="store_true",
        help="Slowly rotate camera azimuth during animation for cinematic effect",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_path = args.model if args.model is not None else _default_model_path()

    run_eval(
        model_path    = model_path,
        n_episodes    = args.episodes,
        device        = args.device,
        seed_base     = args.seed,
        save_video    = not args.no_video,
        all_episodes  = args.all_episodes,
        elev          = args.elev,
        azim          = args.azim,
        rotate        = args.rotate,
    )
