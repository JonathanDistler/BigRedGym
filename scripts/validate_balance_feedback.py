"""Small serial matched native-physics balance comparison; no training or sweeps."""

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np, torch
from experiments.multi_impulse.core import MultiEnvironment, MultiConfig
from experiments.multi_impulse.network import MultiActorCritic
from experiments.multi_impulse.balance import BalanceTeacher
from experiments.balance_feedback.actor import initialize_from_teacher
from experiments.morphology.learning.environment import Assets
from experiments.front_catch.data import raw_state, validate_state, sha
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    prep = Path("output/multi_impulse_preparation_v1")
    checkpoint = Path("output/multi_impulse_study_v2/checkpoint_000050.pt")
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    run = saved["runs"]["B3_seed42"]
    cfg = MultiConfig(**run["config"])
    cfg.kp = 60.0
    cfg.kd = 3.0
    cfg.seconds = 6.2
    cfg.full_trace = True
    assets = Assets(a.output / "assets")
    e = MultiEnvironment(assets, cfg)
    e.reset(1, 0)
    v = np.load(prep / "teacher.npz")
    teacher = BalanceTeacher.__new__(BalanceTeacher)
    for k in ("q", "target", "K", "indices"):
        setattr(teacher, k, v[k])
    actor, meta = initialize_from_teacher(e, teacher)
    baseline = MultiActorCritic(enabled=True)
    baseline.load_state_dict(run["network"])
    baseline.eval()
    torch.save(
        dict(
            network=actor.state_dict(),
            metadata=meta,
            config=vars(cfg),
            teacher_sha256=sha(prep / "teacher.npz"),
            parent_sha256=sha(checkpoint),
        ),
        a.output / "balance_actor.pt",
    )
    bank = torch.load(prep / "balance_bank.pt", weights_only=False, map_location="cpu")
    states = [("held_original", bank["evaluation"]["near_vertical"][0]["state"])]
    # All extra initial states are perturbations of the same valid held-out state,
    # not selected using controller outcomes. No reset occurs within an episode.
    for label, delta_v, delta_pitch in [
        ("pitch_rate_plus", 0.08, 0),
        ("pitch_rate_minus", -0.08, 0),
        ("pitch_offset", 0, 0.003),
    ]:
        e.restore_start(states[0][1], "near_vertical")
        q = e.d.qpos.copy()
        vel = e.d.qvel.copy()
        vel[4] += delta_v
        dq = np.zeros(e.m.nv)
        dq[4] = delta_pitch
        mujoco.mj_integratePos(e.m, q, dq, 1.0)
        good, detail = raw_state(
            e,
            "near_vertical",
            99100 + len(states),
            qpos=q,
            qvel=vel,
            target=e.target.copy(),
        )
        if not good:
            raise RuntimeError(f"Invalid held-out proposal {label}: {detail}")
        states.append((label, e.state()))
    torch.save(states, a.output / "evaluation_states.pt")
    results = []
    render_runs = []
    for label, state in states:
        for name, net in [("B3", baseline), ("observable_LQR", actor)]:
            obs, _ = e.restore_start(state, "near_vertical")
            valid, detail = validate_state(e)
            if not valid:
                raise RuntimeError(detail)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            frames = []

            def record():
                x = np.empty(mujoco.mj_stateSize(e.m, spec))
                mujoco.mj_getState(e.m, e.d, x, spec)
                frames.append(x)

            if label == "held_original":
                record()
            while True:
                with torch.no_grad():
                    batch = torch.tensor(np.repeat(obs[None], 16, axis=0))
                    act = (
                        net.actor(batch).tanh() if name == "B3" else net(batch)
                    ).numpy()[0]
                (obs, _), _, done, info = e.step(act)
                if label == "held_original":
                    record()
                if done:
                    break
            physics = e.physics_trace
            legal = [r for r in physics if r["prefix_safe"]]
            row = dict(
                controller=name,
                reset=label,
                hold=e.multi_best_hold,
                success=bool(e.ever_hold_success),
                seconds=info["episode_seconds"],
                termination=info["termination"],
                illegal_bodies=info["illegal_bodies"],
                collision=bool(info["nonfoot_collision"] or info["self_collision"]),
                max_torque_ratio=info["max_torque_ratio"],
                max_speed_ratio=info["max_speed_ratio"],
                mean_saturated_fraction=float(
                    np.mean([r["saturated_fraction"] for r in physics])
                ),
                com_support_rms=float(
                    np.sqrt(
                        np.mean(
                            [
                                r["com_support_x"] ** 2 + r["com_support_y"] ** 2
                                for r in legal
                            ]
                        )
                    )
                ),
                com_velocity_rms=float(
                    np.sqrt(
                        np.mean(
                            [
                                sum(r["com_velocity_" + k] ** 2 for k in "xyz")
                                for r in legal
                            ]
                        )
                    )
                ),
                angular_velocity_rms=float(
                    np.sqrt(np.mean([r["body_angular_speed"] ** 2 for r in legal]))
                ),
                front_support_fraction=float(
                    np.mean([r["front_support"] for r in physics])
                ),
                max_front_slip_speed=max(r["front_slip_speed"] for r in physics),
            )
            results.append(row)
            print(row, flush=True)
            if label == "held_original":
                save_csv(a.output / f"{name}_trace.csv", e.trace)
                save_csv(a.output / f"{name}_physics.csv", physics)
                np.savez_compressed(
                    a.output / f"{name}_states.npz", integration=np.asarray(frames)
                )
                render_runs.append((name, np.asarray(frames)))
    (a.output / "results.json").write_text(
        json.dumps(
            dict(
                metadata=meta,
                results=results,
                comparison_checkpoint=str(checkpoint),
                reset_scope="four correlated near-vertical states, one original held-out plus fixed perturbations",
            ),
            indent=2,
        )
    )
    if a.render:
        render(e, render_runs, a.output)


def render(e, runs, out):
    from PIL import Image, ImageDraw
    from scripts.render_adc_front_capture import marker

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.47]
    camera.azimuth = 90
    camera.elevation = 0
    camera.distance = 2.0
    images = []
    for name, states in runs:
        renderer = mujoco.Renderer(e.m, height=300, width=400)
        frames = []
        for i, state in enumerate(states):
            mujoco.mj_setState(e.m, e.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(e.m, e.d)
            renderer.update_scene(e.d, camera=camera)
            marker(renderer.scene, e.d.subtree_com[e.base], 0.018, [1, 0, 0, 1])
            for c in e.d.contact:
                if c.dist <= 0:
                    bad = any(
                        b != 0 and b not in e.feet
                        for b in e.m.geom_bodyid[[c.geom1, c.geom2]]
                    )
                    marker(
                        renderer.scene,
                        c.pos,
                        0.018,
                        [1, 0, 0, 1] if bad else [0, 1, 0, 1],
                    )
            im = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 400, 33), fill="black")
            draw.text(
                (4, 3), f"{name} | nominal physics | {i * 0.02:.2f}s", fill="white"
            )
            draw.text(
                (4, 18),
                "Analytical neural feedback" if name != "B3" else "Existing learned B3",
                fill="white",
            )
            frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE))
        renderer.close()
        frames[0].save(
            out / f"{name}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=20,
            loop=0,
        )
        images.append(frames)
    combined = []
    for i in range(max(map(len, images))):
        canvas = Image.new("RGB", (800, 300))
        for j, frames in enumerate(images):
            frame = frames[min(i, len(frames) - 1)].convert("RGB")
            if i >= len(frames):
                ImageDraw.Draw(frame).text(
                    (4, 280), "EPISODE ENDED: final frame held", fill="red"
                )
            canvas.paste(frame, (j * 400, 0))
        combined.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))
    combined[0].save(
        out / "comparison.gif",
        save_all=True,
        append_images=combined[1:],
        duration=20,
        loop=0,
    )
    with Image.open(out / "comparison.gif") as im:
        im.seek(im.n_frames - 1)
        im.load()
        assert im.n_frames > 1
    print("Verified real MuJoCo comparison GIF", out / "comparison.gif", flush=True)


if __name__ == "__main__":
    main()
