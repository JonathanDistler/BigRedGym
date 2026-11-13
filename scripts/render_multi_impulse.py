"""Exact-state, fixed-world-camera MuJoCo replay; one learned actor per condition."""

import argparse, json, os, sys, gc
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np, torch
from PIL import Image, ImageDraw
from experiments.multi_impulse.core import MultiConfig, MultiEnvironment
from experiments.multi_impulse.network import MultiActorCritic
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.environment import Assets
from scripts.render_adc_front_capture import marker
from scripts.morphology_feasibility import save_csv
from scripts.run_com_energy_phase import dump


def collect(checkpoint, mode, seed, kind, preparation, out, teacher_reference=False):
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    run = saved["runs"][mode] if "runs" in saved else saved
    legacy = "actor.0.weight" in run["network"]
    net = (
        ActorCritic(73, 103)
        if legacy
        else MultiActorCritic(enabled=not mode.startswith("B0"))
    )
    net.load_state_dict(run["network"])
    cfg = MultiConfig(**run["config"])
    cfg.kp = 60.0
    cfg.kd = 3.0
    cfg.full_trace = True
    if legacy:
        cfg.context_enabled = False
    env = MultiEnvironment(Assets(out / "assets"), cfg)
    if kind == "standing":
        obs, _ = env.reset(1.0, seed)
    else:
        bank = torch.load(
            preparation / "balance_bank.pt", weights_only=False, map_location="cpu"
        )
        obs, _ = env.restore_start(
            bank["evaluation"]["near_vertical"][0]["state"], "near_vertical"
        )
    teacher = None
    if teacher_reference:
        from experiments.multi_impulse.balance import BalanceTeacher

        values = np.load(preparation / "teacher.npz")
        teacher = BalanceTeacher.__new__(BalanceTeacher)
        teacher.q = values["q"]
        teacher.target = values["target"]
        teacher.K = values["K"]
        teacher.indices = values["indices"]
    states = []
    q = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def record():
        state = np.empty(mujoco.mj_stateSize(env.m, spec))
        mujoco.mj_getState(env.m, env.d, state, spec)
        states.append(state)
        q.append(np.r_[env.d.qpos, env.d.qvel])

    record()
    while True:
        with torch.no_grad():
            action = (
                net.actor(
                    torch.tensor(
                        np.repeat(
                            (obs[:73] if legacy else obs)[None],
                            saved.get("protocol", {}).get("n", 16),
                            axis=0,
                        )
                    )
                )
                .tanh()
                .numpy()[0]
            )
        if teacher is not None:
            action = teacher.command(env)
        (obs, _), _, done, info = env.step(action)
        record()
        if done:
            break
    path = out / f"{mode}_{kind}_seed{seed}"
    np.savez_compressed(
        str(path) + ".npz", integration=states, state=q, time=np.arange(len(q)) * cfg.dt
    )
    save_csv(Path(str(path) + ".csv"), env.trace)
    save_csv(out / f"{path.name}_physics.csv", env.physics_trace)
    info["front_events"] = env.corrections.front_events
    info["teacher_reference"] = teacher_reference
    dump(
        Path(str(path) + ".json"),
        dict(
            **info,
            frames=len(states),
            inference_batch=saved.get("protocol", {}).get("n", 16),
        ),
    )
    return env, np.asarray(states), np.asarray(q), path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=[f"B{i}_seed42" for i in range(4)])
    p.add_argument("--seed", type=int, default=96000)
    p.add_argument(
        "--start-kind", choices=["standing", "near_vertical"], default="standing"
    )
    p.add_argument(
        "--preparation", type=Path, default=Path("output/multi_impulse_preparation_v1")
    )
    p.add_argument(
        "--teacher-reference",
        action="store_true",
        help="Offline validator, explicitly not a learned policy",
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    runs = [
        collect(
            a.checkpoint,
            mode,
            a.seed,
            a.start_kind,
            a.preparation,
            a.output,
            a.teacher_reference,
        )
        for mode in a.modes
    ]
    lo = min(q[:, 0].min() for _, _, q, _ in runs) - 0.6
    hi = max(q[:, 0].max() for _, _, q, _ in runs) + 0.6
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [(lo + hi) / 2, 0, 0.45]
    camera.azimuth = 90
    camera.elevation = 0
    camera.distance = max(1.9, (hi - lo) / (2 * np.tan(np.deg2rad(22.5)) * 480 / 360))
    # Camera framing uses offline extents, never policy observations or control.
    for (env, states, q, path), mode in zip(runs, a.modes):
        renderer = mujoco.Renderer(env.m, height=360, width=480)
        frames = []
        for index, state in enumerate(states):
            mujoco.mj_setState(env.m, env.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(env.m, env.d)
            renderer.update_scene(env.d, camera=camera)
            com = env.d.subtree_com[env.base].copy()
            marker(renderer.scene, com, 0.018, [1, 0, 0, 1])
            for c in env.d.contact:
                if c.dist <= 0:
                    bodies = env.m.geom_bodyid[[c.geom1, c.geom2]]
                    illegal = any(b != 0 and b not in env.feet for b in bodies)
                    marker(
                        renderer.scene,
                        c.pos,
                        0.02,
                        [1, 0, 0, 1] if illegal else [0, 1, 0, 1],
                    )
            im = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(im)
            cam = renderer.scene.camera[0]
            relative = com - cam.pos
            depth = relative @ cam.forward
            focal = 360 * cam.frustum_near / (cam.frustum_top - cam.frustum_bottom)
            if depth > 0:
                cx = 240 + focal * (relative @ np.cross(cam.forward, cam.up)) / depth
                cy = 180 - focal * (relative @ cam.up) / depth
                draw.ellipse(
                    (cx - 3, cy - 3, cx + 3, cy + 3), fill="red", outline="white"
                )
            rotation = env.d.xmat[env.base].reshape(3, 3)
            pitch = np.rad2deg(np.arctan2(-rotation[2, 0], rotation[0, 0]))
            draw.rectangle((0, 0, 480, 33), fill=(15, 15, 15))
            if a.teacher_reference:
                draw.text(
                    (5, 345),
                    "OFFLINE TEACHER REFERENCE - NOT LEARNED POLICY",
                    fill="yellow",
                )
            draw.text(
                (5, 3),
                f"{mode} | {a.start_kind} | nominal | {index * 0.02:.2f}s",
                fill="white",
            )
            draw.text(
                (5, 18),
                f"Pitch {pitch:.1f} deg; COM red; legal contact green",
                fill="white",
            )
            frames.append(im)
        renderer.close()
        gif = Path(str(path) + ".gif")
        frames[0].save(
            gif, save_all=True, append_images=frames[1:], duration=20, loop=0
        )
        assert Image.open(gif).n_frames > 1
        del frames
        gc.collect()
        print("verified native video", gif, flush=True)
    if len(runs) > 1:
        images = [Image.open(str(r[-1]) + ".gif") for r in runs]
        frames = []
        for index in range(max(im.n_frames for im in images)):
            canvas = Image.new("RGB", (480 * len(images), 360))
            for j, im in enumerate(images):
                im.seek(min(index, im.n_frames - 1))
                panel = im.convert("RGB")
                if index >= im.n_frames:
                    ImageDraw.Draw(panel).text(
                        (5, 340), "EPISODE ENDED: final frame held", fill="red"
                    )
                canvas.paste(panel, (j * 480, 0))
            frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))
        path = a.output / "comparison.gif"
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=20, loop=0
        )
        assert Image.open(path).n_frames > 1
        print("verified synchronized comparison", path)


if __name__ == "__main__":
    main()
