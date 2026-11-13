"""Deterministic exact-state MuJoCo side replay and synchronized comparison GIF."""

import argparse
import os
import sys
from pathlib import Path

# EGL works headless; set MUJOCO_GL=osmesa where Mesa is available.
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from experiments.front_catch.core import CatchConfig
from experiments.front_catch.anticipatory import environment_class
from experiments.front_catch.data import network
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from scripts.morphology_feasibility import save_csv
from scripts.run_com_energy_phase import dump


def marker(scene, pos, size, color):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size] * 3),
        np.asarray(pos),
        np.eye(3).ravel(),
        np.array(color, dtype=np.float32),
    )
    scene.ngeom += 1


def replay(
    checkpoint,
    mode,
    out,
    seed=96000,
    kind="standing",
    preparation=None,
    width=480,
    height=360,
):
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if "runs" in saved:
        run = saved["runs"][mode]
        net = ActorCritic(73, 103)
        net.load_state_dict(run["network"])
        cfg = CatchConfig(**run["config"])
    else:
        net, run = network(checkpoint)
        from experiments.front_catch.data import config

        cfg = config(run, "B")
        cfg.comparison_suite = "anticipatory_v1"
    # All videos and comparisons use original Go2 dynamics.
    protocol = saved.get("protocol", {})
    cfg.kp = protocol.get("nominal_k", 60.0)
    cfg.kd = protocol.get("nominal_b", 3.0)
    cfg.full_trace = True
    env = environment_class(cfg)(Assets(out / "assets"), cfg)
    if kind == "standing":
        obs, _ = env.reset(1.0, seed)
    else:
        bank = torch.load(
            preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
        )
        obs, _ = env.restore_start(bank["evaluation"][kind][0]["state"], kind)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.12, 0, 0.40]
    camera.distance = 1.9
    camera.azimuth = 90
    camera.elevation = 0
    # Never change the physical model; this only configures the renderer buffer.
    env.m.vis.global_.offwidth = max(width, env.m.vis.global_.offwidth)
    env.m.vis.global_.offheight = max(height, env.m.vis.global_.offheight)
    renderer = mujoco.Renderer(env.m, height=height, width=width)
    frames = []
    states = []
    times = []
    targets = []
    rows = []
    integration = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def frame(row=None):
        renderer.update_scene(env.d, camera=camera)
        scene = renderer.scene
        com = env.d.subtree_com[env.base].copy()
        marker(scene, com, 0.018, [1, 0.2, 0.1, 1])
        feet = env.d.geom_xpos[env.foot_geoms[:2]].copy()
        loaded = env.row["loads"][:2] > 0.05 * env.weight
        for i in range(2):
            marker(
                scene,
                feet[i],
                0.025,
                [0.1, 1, 0.2, 1] if loaded[i] else [0.2, 0.4, 1, 1],
            )
        if loaded.all():
            for t in np.linspace(0, 1, 15):
                marker(scene, (1 - t) * feet[0] + t * feet[1], 0.007, [0.2, 1, 0.7, 1])
        support_contacts = []
        for index, contact in enumerate(env.d.contact):
            if contact.dist <= 0:
                bodies = env.m.geom_bodyid[[contact.geom1, contact.geom2]]
                illegal = any(b != 0 and b not in env.feet for b in bodies)
                wrench = np.zeros(6)
                mujoco.mj_contactForce(env.m, env.d, index, wrench)
                if (
                    0 in bodies
                    and any(b in env.feet[:2] for b in bodies)
                    and wrench[0] > 0
                ):
                    support_contacts.append(contact.pos.copy())
                marker(
                    scene,
                    contact.pos,
                    0.03 if illegal else 0.012,
                    [1, 0, 0, 1] if illegal else [1, 1, 0, 1],
                )
        im = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(im)
        # Screen overlay keeps the actual COM visible through opaque robot links.
        gl_camera = scene.camera[0]
        forward = np.asarray(gl_camera.forward)
        up = np.asarray(gl_camera.up)
        right = np.cross(forward, up)
        relative = com - gl_camera.pos
        depth = float(relative @ forward)
        focal = (
            height
            * gl_camera.frustum_near
            / (gl_camera.frustum_top - gl_camera.frustum_bottom)
        )
        if depth > 0:
            cx = width / 2 + focal * float(relative @ right) / depth
            cy = height / 2 - focal * float(relative @ up) / depth
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="red", outline="white")
        if env.trace:
            # Native active contact interval, separately from inflated foot-region
            # distance used in the legacy and shaped objectives.
            points = np.asarray(support_contacts).reshape(-1, 3)
            lo = float(points[:, 0].min()) if len(points) else float("nan")
            hi = float(points[:, 0].max()) if len(points) else float("nan")
            env.trace[-1].update(
                native_support_min_x=lo,
                native_support_max_x=hi,
                native_com_outside_interval_x=max(lo - com[0], com[0] - hi, 0)
                if len(points)
                else float("nan"),
            )
        draw.rectangle((0, 0, width, 33), fill=(20, 20, 20))
        draw.text(
            (5, 3), f"{mode} | nominal Go2 | {kind} | t={env.d.time:.3f}s", fill="white"
        )
        draw.text(
            (5, 18),
            f"pitch={np.rad2deg(env.row['pitch']):.1f} deg  "
            "COM:red contact:yellow impact:red",
            fill="white",
        )
        frames.append(im)
        states.append(np.r_[env.d.qpos, env.d.qvel])
        times.append(float(env.d.time))
        targets.append(env.target.copy())
        state = np.empty(mujoco.mj_stateSize(env.m, spec))
        mujoco.mj_getState(env.m, env.d, state, spec)
        integration.append(state)

    frame()
    while True:
        with torch.no_grad():
            # Match the evaluation GEMM/tanh batch shape; long sliding contact
            # sequences can amplify float-rounding differences from scalar GEMV.
            batch = np.repeat(obs[None], protocol.get("num_envs", 16), axis=0)
            action = net.actor(torch.tensor(batch)).tanh().numpy()[0]
        (obs, _), _, done, info = env.step(action)
        frame()
        rows.append(dict(env.trace[-1], replay_seed=seed))
        if done:
            break
    renderer.close()
    path = out / f"{mode}_{kind}_seed{seed}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(cfg.dt * 1000),
        loop=0,
    )
    # Persist full floating-base state and integrator state; CSV is never used to
    # reconstruct geometry. Physics trace includes 500-Hz contact-force samples.
    np.savez_compressed(
        path.with_suffix(".npz"),
        time=times,
        state=states,
        target=targets,
        integration=integration,
    )
    save_csv(path.with_suffix(".csv"), rows)
    save_csv(out / f"{mode}_{kind}_seed{seed}_physics.csv", env.physics_trace)
    dump(
        path.with_suffix(".json"),
        dict(**info, frames=len(frames), seed=seed, nominal_k=cfg.kp, nominal_b=cfg.kd),
    )
    # Verify an actual decodable, changing robot animation was produced.
    check = Image.open(path)
    if check.n_frames < 2 or np.array_equal(
        np.asarray(frames[0]), np.asarray(frames[-1])
    ):
        raise RuntimeError("Empty/static render")
    print("verified video", path, "frames", check.n_frames, flush=True)
    return frames


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--modes", nargs="+", default=["A_seed42", "B_seed42", "C_seed42", "D_seed42"]
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=96000)
    p.add_argument(
        "--start-kind",
        choices=[
            "standing",
            "loaded",
            "pre_liftoff",
            "post_liftoff",
            "partial",
            "near_vertical",
        ],
        default="standing",
    )
    p.add_argument("--preparation", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.start_kind != "standing" and a.preparation is None:
        p.error("Intermediate replay requires preparation")
    videos = [
        replay(a.checkpoint, mode, a.output, a.seed, a.start_kind, a.preparation)
        for mode in a.modes
    ]
    if len(videos) > 1:
        combined = []
        for t in range(max(map(len, videos))):
            canvas = Image.new("RGB", (480 * len(videos), 360))
            for i, frames in enumerate(videos):
                panel = frames[min(t, len(frames) - 1)].copy()
                if t >= len(frames):
                    ImageDraw.Draw(panel).text(
                        (10, 340), "EPISODE ENDED (frozen final frame)", fill="red"
                    )
                canvas.paste(panel, (480 * i, 0))
            combined.append(canvas)
        path = a.output / "comparison.gif"
        combined[0].save(
            path, save_all=True, append_images=combined[1:], duration=20, loop=0
        )
        if Image.open(path).n_frames < 2:
            raise RuntimeError("Invalid comparison")
        print("verified comparison", path, flush=True)


if __name__ == "__main__":
    main()
