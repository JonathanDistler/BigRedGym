"""Distill, evaluate, render, benchmark and resume ONE Go2 neural actor."""

import argparse, copy, json, os, sys, time
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
    WANDB_DISABLED="true",
)
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np, torch
from experiments.unified_go2.core import (
    load,
    digest,
    save_json,
    Environment,
    Student,
    Vector,
    promote,
    source_contract,
    adc_network,
)
from experiments.multi_impulse.core import MultiConfig
from experiments.balance_feedback.actor import BalanceActor
from experiments.morphology.learning.environment import Assets
from experiments.front_catch.learning import collect, hp_defaults
from experiments.front_catch.io import atomic_checkpoint, append_csv
from experiments.multi_impulse.learning import safe_update


def setup(prep, out):
    manifest = json.loads((prep / "manifest.json").read_text())
    for name, h in manifest.items():
        if digest(prep / name) != h:
            raise ValueError("Preparation changed: " + name)
    cfg = MultiConfig(**json.loads((prep / "config.json").read_text()))
    cfg.seconds = 12.0
    cfg.full_trace = True
    data = np.load(prep / "demonstrations.npz")
    bank = load(prep / "bank.pt")
    assets = Assets(out / "assets")
    return cfg, data, bank, assets


def actions(net, obs, kind="student"):
    with torch.no_grad():
        x = torch.tensor(obs)
        return (net(x) if kind == "balance" else net.actor(x).tanh()).numpy()


def evaluate(
    net, cfg, bank, assets, centers, controller="student", render_out=None, quick=False
):
    e = Environment(assets, copy.deepcopy(cfg))
    e.capture_centers = centers
    specs = [("standing", s) for s in ([96000] if quick else [96000, 96001])]
    specs += [("bridge", s) for s in bank["evaluation"]["bridge"][: 1 if quick else 2]]
    specs += [
        ("balance", s) for s in bank["evaluation"]["balance"][: 2 if quick else 4]
    ]
    rows = []
    for index, (kind, start) in enumerate(specs):
        obs, _ = (
            e.reset(1.0, start)
            if kind == "standing"
            else e.restore_start(
                start, "near_vertical" if kind == "balance" else "partial"
            )
        )
        frames = []
        minimum = float("inf")
        entered = False
        time_near = 0.0

        def record():
            x = np.empty(mujoco.mj_stateSize(e.m, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(e.m, e.d, x, mujoco.mjtState.mjSTATE_INTEGRATION)
            frames.append(x)

        want = render_out is not None and (
            kind == "standing" or (kind == "balance" and index == len(specs) - 2)
        )
        if want:
            record()
        while True:
            # Batch convention is fixed at one during every deterministic evaluation.
            (obs, _), _, done, info = e.step(actions(net, obs[None], controller)[0])
            minimum = min(minimum, e.capture_distance)
            entered |= e.capture_distance < 1
            time_near += 0.02 * float(e.capture_distance < 1)
            if want:
                record()
            if done:
                break
        row = {
            k: info[k]
            for k in [
                "certified_standing_takeoff",
                "peak_collision_free_pitch",
                "handstand_seconds",
                "standing_success",
                "success",
                "episode_seconds",
                "termination",
                "front_support_fraction",
                "nonfoot_collision",
                "self_collision",
                "genuine_pushes",
                "counted_front_recontacts",
                "seconds_above45",
                "seconds_above60",
                "seconds_above75",
                "max_torque_ratio",
                "max_speed_ratio",
            ]
        }
        row.update(
            distribution=kind,
            controller=controller,
            index=index,
            success=bool(e.ever_hold_success),
            capture_distance_min=minimum,
            entered_empirical_capture_neighborhood=bool(entered),
            seconds_in_empirical_neighborhood=time_near,
        )
        rows.append(row)
        if want:
            render_out.mkdir(parents=True, exist_ok=True)
            from scripts.morphology_feasibility import save_csv

            name = f"{controller}_{kind}_{index}"
            save_csv(render_out / (name + "_trace.csv"), e.trace)
            save_csv(render_out / (name + "_physics.csv"), e.physics_trace)
            np.savez_compressed(render_out / (name + "_states.npz"), integration=frames)
            render_episode(
                e, np.array(frames), render_out / (name + ".gif"), controller
            )
    return rows


def score(rows):
    groups = {
        k: [r for r in rows if r["distribution"] == k]
        for k in ["balance", "standing", "bridge"]
    }

    def mean(k, key):
        return float(np.mean([r[key] for r in groups[k]])) if groups[k] else 0.0

    return [
        mean("balance", "success"),
        mean("standing", "standing_success"),
        mean("standing", "certified_standing_takeoff"),
        mean("bridge", "handstand_seconds"),
        mean("standing", "peak_collision_free_pitch"),
    ]


def distill(a, cfg, data, assets, bank):
    torch.manual_seed(a.seed)
    net = Student.from_adc(
        load(a.preparation / "adc.pt"), load(a.preparation / "balance.pt")
    )
    for param in net.actor.affine.parameters():
        param.requires_grad_(False)
    opt = torch.optim.Adam(net.actor.residual.parameters(), lr=3e-4)
    obs = torch.tensor(data["observations"])
    target = torch.tensor(data["actions"])
    role = torch.tensor(data["roles"])
    groups = [torch.where(role == i)[0] for i in (0, 1)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("Both competent teachers required")
    expert = BalanceActor()
    expert.load_state_dict(load(a.preparation / "balance.pt")["network"])
    losses = []
    for step in range(a.steps):
        ids = torch.cat([g[torch.randint(len(g), (64,))] for g in groups])
        x = obs[ids]
        y = target[ids]
        error = (net.actor.position_targets(x) - y).square().mean()
        # Preserve local velocity-feedback sensitivity at actual competent balance states.
        xb = x[64:]
        direction = torch.zeros_like(xb)
        direction[:, 4] = 0.5
        direction[:, 56] = 0.2
        eps = 0.002
        ds = (
            net.actor(xb + eps * direction).tanh()
            - net.actor(xb - eps * direction).tanh()
        ) / (2 * eps)
        with torch.no_grad():
            dt = (expert(xb + eps * direction) - expert(xb - eps * direction)) / (
                2 * eps
            )
        derivative = (ds - dt).square().mean()
        loss = error + 0.1 * derivative
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            losses.append(
                dict(
                    step=step,
                    action_mse=float(error),
                    feedback_derivative_mse=float(derivative),
                )
            )
            print(losses[-1], flush=True)
    atomic_checkpoint(
        a.output / "student.pt",
        dict(
            network=net.state_dict(),
            distillation_optimizer=opt.state_dict(),
            steps=a.steps,
            seed=a.seed,
            preparation_sha=digest(a.preparation / "manifest.json"),
            architecture="single affine-feedback + quadratic-gravity-residual actor93→12, MLP critic123",
            source_contract=source_contract(),
        ),
    )
    save_json(
        a.output / "distillation.json",
        dict(
            losses=losses,
            training_frames=[len(g) for g in groups],
            supervision="Only certified teacher trajectories; derivative preservation at successful balance states; no intermediate teacher blend",
        ),
    )


def train(a, cfg, data, bank, assets):
    restored = load(a.resume) if a.resume else None
    contract = source_contract()
    runtime = dict(
        torch=str(torch.__version__), numpy=np.__version__, mujoco=mujoco.__version__
    )
    if restored:
        if restored["sources"] != contract or restored["runtime"] != runtime:
            raise ValueError("Exact resume requires identical source/runtime")
        if restored["preparation_sha"] != digest(a.preparation / "manifest.json"):
            raise ValueError("Preparation changed")
        if a.n != restored["n"] or a.horizon != restored["horizon"]:
            raise ValueError("Resume cannot change rollout semantics")
        latest = max(
            int(p.stem.split("_")[-1]) for p in a.output.glob("checkpoint_*.pt")
        )
        if (
            a.resume.parent.resolve() != a.output.resolve()
            or restored["iteration"] != latest
        ):
            raise ValueError("Use latest checkpoint in this output")
    net = Student()
    net.load_state_dict((restored or load(a.initial))["network"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    v = Vector(assets, cfg, bank, a.n, data["centers"], a.seed)
    iteration = 0
    steps = 0
    reference = 0.0
    history = []
    best = []
    if restored:
        opt.load_state_dict(restored["optimizer"])
        obs, critic = v.restore(restored["vector"])
        iteration = restored["iteration"]
        steps = restored["steps"]
        reference = restored["reference"]
        history = restored["history"]
        best = restored["best"]
        assert np.array_equal(obs, restored["obs"]) and np.array_equal(
            critic, restored["critic"]
        ), "Inexact observation restore"
        torch.set_rng_state(restored["torch_rng"])
        np.random.set_state(restored["numpy_rng"])
    else:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        obs, critic = v.reset()
    hp = hp_defaults()
    hp.update(bc_weight=1.0, bc_latent_scale=1.0, entropy_weight=0.001)
    weights = np.ones((len(data["roles"]), 12), np.float32)
    weights[data["roles"] == 0, :6] = 0.15
    demonstrations = (
        torch.tensor(data["observations"]),
        torch.tensor(data["actions"]),
        torch.tensor(weights),
    )
    if (a.updates * a.n * a.horizon) > a.interaction_budget:
        raise ValueError("Explicit absolute interaction budget exceeded")
    session = a.output / f"session_{len(list(a.output.glob('session_*'))):04}"
    session.mkdir()

    def save():
        payload = dict(
            network=net.state_dict(),
            optimizer=opt.state_dict(),
            vector=v.state(),
            iteration=iteration,
            steps=steps,
            n=a.n,
            horizon=a.horizon,
            obs=obs,
            critic=critic,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            reference=reference,
            history=history,
            best=best,
            sources=contract,
            runtime=runtime,
            preparation_sha=digest(a.preparation / "manifest.json"),
            hp=hp,
        )
        path = a.output / f"checkpoint_{iteration:06}.pt"
        atomic_checkpoint(path, payload)
        save_json(
            a.output / "latest.json", dict(checkpoint=path.name, iteration=iteration)
        )
        return path

    if not restored:
        save()
    start = time.monotonic()
    while iteration < a.updates:
        if (a.output / "STOP").exists() or (
            a.max_seconds and time.monotonic() - start >= a.max_seconds
        ):
            break
        batch, obs, critic, episodes, _ = collect(
            net, v, "unified", obs, critic, a.horizon, None, hp
        )
        stats = safe_update(net, opt, batch, hp, demonstrations)
        iteration += 1
        steps += a.n * a.horizon
        if a.eval_every and iteration % a.eval_every == 0:
            # Evaluation uses a separate world and restores RNG, so it cannot perturb PPO.
            rng = torch.get_rng_state()
            nr = np.random.get_state()
            rows = evaluate(net, cfg, bank, assets, data["centers"], quick=True)
            torch.set_rng_state(rng)
            np.random.set_state(nr)
            current = score(rows)
            reference = max(reference, current[0])
            v.stage = promote(v.stage, rows, reference)
            history.extend(dict(iteration=iteration, **r) for r in rows)
            if not best or current > best:
                best = current
                save_json(
                    a.output / "best.json",
                    dict(
                        iteration=iteration,
                        score=current,
                        checkpoint=f"checkpoint_{iteration:06}.pt",
                    ),
                )
            save_json(session / f"evaluation_{iteration:06}.json", rows)
        append_csv(
            session / "training.csv",
            [dict(iteration=iteration, steps=steps, stage=v.stage, **stats)],
        )
        if episodes:
            save_json(session / f"episodes_{iteration:06}.json", episodes)
        path = save()
        print("saved", path, "KL", stats["post_update_kl"], flush=True)
    save_json(
        session / "status.json",
        dict(
            iteration=iteration,
            steps=steps,
            target=a.updates,
            completed=iteration == a.updates,
        ),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=["distill", "train", "evaluate", "benchmark"])
    p.add_argument(
        "--preparation", type=Path, default=Path("output/unified_go2_preparation_v1")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--interaction-budget", type=int, default=100000)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--controllers", nargs="+", default=["student"])
    p.add_argument("--render", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if not 1 <= a.n <= 16:
        p.error("Serial env count must be1–16; no subprocess workers")
    if a.resume:
        if not a.output.exists():
            p.error("Resume requires original output")
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(a.preparation, a.output)
    if a.command == "distill":
        distill(a, cfg, data, assets, bank)
    elif a.command == "train":
        train(a, cfg, data, bank, assets)
    elif a.command == "evaluate":
        rows = []
        for c in a.controllers:
            if c == "balance":
                net = BalanceActor()
                net.load_state_dict(load(a.preparation / "balance.pt")["network"])
            elif c == "adc":
                net = adc_network(load(a.preparation / "adc.pt"))
            else:
                net = Student()
                net.load_state_dict(load(a.initial)["network"])
            rows.extend(
                evaluate(
                    net,
                    cfg,
                    bank,
                    assets,
                    data["centers"],
                    c,
                    a.output / "videos" if a.render else None,
                    a.quick,
                )
            )
        save_json(a.output / "evaluation.json", rows)
        print(
            "Evaluation",
            [
                (
                    r["controller"],
                    r["distribution"],
                    round(r["handstand_seconds"], 3),
                    r["success"],
                )
                for r in rows
            ],
            flush=True,
        )
    else:
        from resource import getrusage, RUSAGE_SELF

        measurements = []
        for n in [2, 4, 8]:
            v = Vector(assets, cfg, bank, n, data["centers"])
            net = Student.from_adc(
                load(a.preparation / "adc.pt"), load(a.preparation / "balance.pt")
            )
            obs, critic = v.reset()
            t = time.monotonic()
            collect(net, v, "unified", obs, critic, 4, None, hp_defaults())
            measurements.append(
                dict(
                    n=n,
                    workers=0,
                    steps_per_second=4 * n / (time.monotonic() - t),
                    peak_rss_kib=getrusage(RUSAGE_SELF).ru_maxrss,
                )
            )
            del v
        save_json(a.output / "benchmark.json", measurements)
        print(measurements)


def render_episode(env, states, path, label):
    from PIL import Image, ImageDraw
    from scripts.render_adc_front_capture import marker

    renderer = mujoco.Renderer(env.m, height=300, width=400)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = 0
    camera.lookat[:] = [0, 0, 0.45]
    camera.distance = 2.8
    frames = []
    for index, state in enumerate(states):
        mujoco.mj_setState(env.m, env.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(env.m, env.d)
        renderer.update_scene(env.d, camera=camera)
        marker(renderer.scene, env.d.subtree_com[env.base], 0.018, [1, 0, 0, 1])
        for contact in env.d.contact:
            if contact.dist <= 0:
                illegal = any(
                    b != 0 and b not in env.feet
                    for b in env.m.geom_bodyid[[contact.geom1, contact.geom2]]
                )
                marker(
                    renderer.scene,
                    contact.pos,
                    0.018,
                    [1, 0, 0, 1] if illegal else [0, 1, 0, 1],
                )
        frame = Image.fromarray(renderer.render().copy())
        ImageDraw.Draw(frame).text(
            (4, 4), f"{label} | nominal | {index * 0.02:.2f}s", fill="white"
        )
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
    renderer.close()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=20, loop=0)
    with Image.open(path) as image:
        image.seek(image.n_frames - 1)
        image.load()
        assert image.n_frames > 1


if __name__ == "__main__":
    main()
