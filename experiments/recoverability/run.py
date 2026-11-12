"""Phases 1–3 only: inventory, compatibility, empirical recovery, frozen switching.

No optimizer, PPO update, or extended-training command is exposed.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import zipfile
import mujoco
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.controllers import (
    SavedController,
    BalanceAdapter,
    CANDIDATES,
)
from experiments.recoverability.physics import World, Metrics, feasible_state

KICK = Path("output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt")


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(set().union(*(r.keys() for r in rows)))
        )
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    if n == 0:
        return [0.0, 1.0]
    z = 1.96
    d = 1 + z * z / n
    c = (k / n + z * z / (2 * n)) / d
    h = z * np.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def source_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    manifest = Path("recoverability_bundle_manifest.json")
    return (
        json.loads(manifest.read_text())["base_commit"]
        if manifest.exists()
        else "unversioned: inspect source hashes"
    )


def inventory(out):
    paths = []
    for group in (
        "go2balance",
        "go2balance_longhold",
        "go2backwards_reachability",
        "go2reachability_fast_probe",
    ):
        for p in (Path("logs") / group).glob("*/model_*.pt"):
            paths.append(
                dict(
                    path=str(p),
                    sha256=sha(p),
                    bytes=p.stat().st_size,
                    saved_sources=len(list((p.parent / "files").rglob("*.py"))),
                )
            )
    archives = []
    for p in Path("downloaded_models").glob("*balance*.zip"):
        checks = []
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.endswith(".pt"):
                    continue
                expected = hashlib.sha256(z.read(name)).hexdigest()
                checks.append(
                    dict(
                        member=name,
                        sha256=expected,
                        local_match=Path(name).is_file() and sha(name) == expected,
                    )
                )
        archives.append(dict(path=str(p), sha256=sha(p), checkpoints=checks))
    dump(out / "inventory.json", dict(checkpoints=paths, archives=archives))
    return paths


def model_audit(c, w):
    native = c.env._backend._model_for_env(0)
    m = w.m
    fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_condim",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "jnt_range",
        "dof_armature",
        "dof_damping",
    )
    differences = {}
    for key in fields:
        a, b = getattr(native, key), getattr(m, key)
        differences[key] = (
            float(np.max(abs(a - b))) if a.shape == b.shape else "shape_mismatch"
        )
    e = c.env
    gains = dict(
        kp=e.p_gains[0].tolist(),
        kd=e.d_gains[0].tolist(),
        ctrl_dt=e.dt,
        physics_dt=native.opt.timestep,
    )
    # Different physical model or gains cannot be hidden by the observation adapter.
    compatible = all(isinstance(v, float) and v < 1e-8 for v in differences.values())
    compatible &= bool(
        np.allclose(gains["kp"], w.cfg.kp) and np.allclose(gains["kd"], w.cfg.kd)
    )
    compatible &= (
        abs(native.opt.timestep - m.opt.timestep) < 1e-10 and abs(e.dt - 0.01) < 1e-8
    )
    return dict(
        compatible=compatible,
        differences=differences,
        gains=gains,
        mass=float(m.body_mass.sum()),
        native_mass=float(native.body_mass.sum()),
        friction=m.geom_friction[0].tolist(),
        joint_names=w.names,
        effort=w.effort.tolist(),
        speed=w.speed.tolist(),
        limits=w.bounds.tolist(),
        source_checkpoint=str(c.path),
        adaptation=(
            "Separate saved-source encoder and residual-to-absolute target; "
            "100 Hz balance, 50 Hz kick; shared 500 Hz physics and motor limits"
        ),
    )


def rollout(c, assets, cfg, seed, parameters, trace_path=None, collect=False):
    w = World(assets, cfg)
    reference = c.env._standing_posture()[0].numpy()
    accepted, screen = feasible_state(w, reference, seed, **parameters)
    if not accepted:
        return dict(seed=seed, accepted=False, screen=screen, success=False), [], []
    adapter = BalanceAdapter(c, w)
    m = Metrics(w)
    initial = m.observe(w.physical_row(), 0.0)
    trace = []
    bank = []
    target = w.target.copy()
    for tick in range(round(6.5 / w.m.opt.timestep)):
        if tick % 5 == 0:
            target = adapter.target()
        row = w.tick(target)
        adapter.advance_memory(row["loads"], w.m.opt.timestep)
        sample = m.observe(row, w.m.opt.timestep)
        if tick % 10 == 0:
            trace.append(sample)
        if collect and tick in (49, 149, 249):
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(w.m, spec))
            mujoco.mj_getState(w.m, w.d, state, spec)
            bank.append(
                dict(
                    features=sample.copy(),
                    integration=state.copy(),
                    target=target.copy(),
                    phase=adapter.phase.numpy().copy() if c.dim == 59 else np.zeros(0),
                    filtered_load=adapter.load.numpy().copy()
                    if c.dim == 59
                    else np.zeros(0),
                )
            )
        if m.failed:
            break
    if trace_path:
        csv_write(trace_path, trace)
    return (
        dict(seed=seed, accepted=True, screen=screen, initial=initial, **m.result()),
        trace,
        bank,
    )


CASES = [
    ("nominal", {}),
    ("pitch_minus", dict(pitch=88.0)),
    ("pitch_plus", dict(pitch=92.0)),
    ("roll_minus", dict(roll=-2.0)),
    ("roll_plus", dict(roll=2.0)),
    ("omega_minus", dict(omega=-0.25)),
    ("omega_plus", dict(omega=0.25)),
    ("front_contact", dict(foot_height=0.004)),
    ("rear_config", dict(rear_delta=-0.08)),
]


def evaluate_candidates(assets, cfg, out, seeds, roll_only=False):
    rows = []
    compat = []
    for label, path, task in CANDIDATES:
        c = None
        try:
            c = SavedController(path, task)
            audit = model_audit(c, World(assets, cfg))
            dump(
                out / f"{label}_configuration.json",
                dict(controller=c.meta, physics=audit),
            )
            compat.append(dict(label=label, **audit))
            if not audit["compatible"]:
                continue
            for case, parameters in CASES:
                if roll_only and not case.startswith("roll_"):
                    continue
                for seed in seeds:
                    result, _, _ = rollout(
                        c,
                        assets,
                        cfg,
                        seed,
                        parameters,
                        out / f"{label}_{case}_trace.csv" if seed == seeds[0] else None,
                    )
                    rows.append(dict(controller=label, case=case, **result))
                subset = [
                    r for r in rows if r["controller"] == label and r["case"] == case
                ]
                print(
                    label,
                    case,
                    sum(r["success"] for r in subset),
                    "/",
                    len(subset),
                    flush=True,
                )
        except Exception as error:
            compat.append(dict(label=label, compatible=False, error=repr(error)))
            print("INCOMPATIBLE", label, repr(error), flush=True)
        finally:
            if c:
                c.close()
        dump(out / "candidate_results.json", rows)
        dump(out / "compatibility.json", compat)
    summaries = []
    for label, _, _ in CANDIDATES:
        rs = [r for r in rows if r["controller"] == label]
        if not rs:
            continue
        nominal = [r for r in rs if r["case"] == "nominal"]
        n = sum(r["accepted"] for r in rs)
        k = sum(r["success"] for r in rs)
        summaries.append(
            dict(
                controller=label,
                successes=k,
                total=len(rs),
                accepted=n,
                unconditional_success=k / len(rs),
                conditional_success=k / max(n, 1),
                wilson95=wilson(k, n),
                nominal_success=sum(r["success"] for r in nominal)
                / max(len(nominal), 1),
            )
        )
    dump(out / "candidate_summary.json", summaries)
    eligible = [
        r
        for r in summaries
        if r["nominal_success"] >= 0.8 and r["conditional_success"] >= 0.5
    ]
    best = (
        max(eligible, key=lambda x: (x["conditional_success"], x["nominal_success"]))
        if eligible
        else None
    )
    dump(
        out / "balance_gate.json",
        dict(
            suitable=best is not None,
            selected=best,
            criterion=(
                ">=80% nominal and >=50% aggregate success on feasible proposals; "
                "not a global robustness guarantee"
            ),
        ),
    )
    local = [r for r in summaries if r["nominal_success"] >= 0.8]
    nominal_best = max(local, key=lambda x: x["conditional_success"]) if local else None
    dump(
        out / "reference_gate.json",
        dict(
            selected=nominal_best,
            broad_perturbation_gate_passed=best is not None,
            scope=(
                "Nominal five-second reference; map recovery before "
                "switching or transition training"
            ),
        ),
    )
    return nominal_best


FEATURES = (
    [
        "pitch",
        "omega",
        "roll_rate",
        "com_support_x",
        "com_z",
        "com_vx",
        "com_vz",
        "force_FL",
        "force_FR",
        "drift",
    ]
    + [f"q{j}" for j in range(12)]
    + [f"v{j}" for j in range(12)]
)
SCALES = np.array(
    [10.0, 1.0, 1.0, 0.05, 0.10, 0.5, 0.5, 80.0, 80.0, 0.04] + [0.3] * 12 + [2.0] * 12
)


def feature(row):
    return np.array([row[k] for k in FEATURES], dtype=float) / SCALES


def maps(c, assets, cfg, out, seeds, fine=False):
    rows = []
    successful = []
    states = []
    cells = []
    proposals = []
    angles = (
        (88.0, 89.0, 89.5, 90.0, 90.5, 91.0, 92.0)
        if fine
        else (60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0)
    )
    rates = (-0.3, -0.15, 0.0, 0.15, 0.3) if fine else (-0.75, -0.25, 0.0, 0.25, 0.75)
    for pitch in angles:
        for omega in rates:
            proposals.append(
                ("pitch_omega", pitch, omega, dict(pitch=pitch, omega=omega))
            )
    for delta in () if fine else (-0.10, -0.05, 0.0, 0.05, 0.10):
        proposals.append(("front_config", delta, 0.0, dict(front_delta=delta)))
    for shift in () if fine else (-0.04, -0.02, 0.0, 0.02, 0.04):
        proposals.append(("com_shift", shift, 0.0, dict(com_shift=shift)))
    for index, (family, x, y, params) in enumerate(proposals):
        trials = []
        banks = []
        for seed in seeds:
            result, _, bank = rollout(c, assets, cfg, seed, params, collect=True)
            rows.append(dict(family=family, x=x, y=y, **result))
            trials.append(result)
            banks.append(bank)
        accepted = sum(r["accepted"] for r in trials)
        successes = sum(r["success"] for r in trials)
        cell = dict(
            family=family,
            x=x,
            y=y,
            trials=len(trials),
            accepted=accepted,
            successes=successes,
            probability=successes / accepted if accepted else None,
            wilson95=wilson(successes, accepted),
        )
        cell["validated"] = (
            accepted >= 6 and successes / accepted >= 0.8 and cell["wilson95"][0] >= 0.5
        )
        cells.append(cell)
        if cell["validated"]:
            for result, bank in zip(trials, banks):
                if result["success"]:
                    for snap in bank:
                        successful.append(
                            dict(
                                cell=index,
                                seed=result["seed"],
                                features=snap["features"],
                            )
                        )
                        states.append(snap)
        print(
            "map",
            family,
            x,
            y,
            successes,
            "/",
            accepted,
            "validated",
            cell["validated"],
            flush=True,
        )
        dump(out / "recovery_cells.json", cells)
        dump(out / "recovery_trials.json", rows)
    dump(out / "validated_region.json", successful)
    entries = {}
    for i, s in enumerate(states):
        for k in ("integration", "target", "phase", "filtered_load"):
            entries[f"{i}_{k}"] = s[k]
    np.savez_compressed(out / "recoverable_states.npz", **entries)
    return cells, successful


def counterfactual_recovery(c, w, assets, cfg, seconds=6.5):
    """Validate a candidate in a scratch rollout without writing live state."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(w.m, spec))
    mujoco.mj_getState(w.m, w.d, state, spec)
    q, v, t = w.d.qpos.copy(), w.d.qvel.copy(), float(w.d.time)
    scratch = World(assets, cfg)
    mujoco.mj_setState(scratch.m, scratch.d, state, spec)
    mujoco.mj_forward(scratch.m, scratch.d)
    scratch.target = w.target.copy()
    adapter = BalanceAdapter(c, scratch)
    metrics = Metrics(scratch)
    if getattr(w, "monitor", None) is not None:
        metrics.anchor = w.monitor.anchor.copy()
        metrics.drift = w.monitor.drift
        metrics.failed = w.monitor.failed
    for tick in range(round(seconds / scratch.m.opt.timestep)):
        if tick % 5 == 0:
            target = adapter.target()
        row = scratch.tick(target)
        adapter.advance_memory(row["loads"], scratch.m.opt.timestep)
        metrics.observe(row, scratch.m.opt.timestep)
        if metrics.failed or metrics.best_hold >= 5.0:
            break
    assert np.array_equal(q, w.d.qpos) and np.array_equal(v, w.d.qvel) and t == w.d.time
    return metrics.result()


def switching(net, c, assets, cfg, out, seeds, region):
    points = (
        np.array([feature(r["features"]) for r in region])
        if region
        else np.empty((0, len(FEATURES)))
    )
    results = []
    for stochastic in (False, True):
        for seed in seeds:
            w = World(assets, cfg)
            actor, _ = w.reset(1.0, seed)
            rng = torch.Generator().manual_seed(seed + 1000000)
            m = Metrics(w)
            m.require_takeoff = True
            w.monitor = m
            trace = []
            distance = float("inf")
            adapter = None
            entry = None
            queries = []
            for control in range(400):
                if adapter is None:
                    with torch.no_grad():
                        dist = net.distribution(torch.tensor(actor))
                        raw = dist.loc
                        if stochastic:
                            raw = raw + dist.scale * torch.randn(12, generator=rng)
                    (actor, _), _, done, info = w.step(raw.tanh().numpy())
                    sample = w.last_monitor_sample
                    if len(points):
                        delta = np.max(abs(points - feature(sample)), axis=1)
                        distance = min(distance, float(delta.min()))
                        # An envelope proposes candidates.
                        # Exact-state forward validation remains mandatory.
                        nearest = int(np.argmin(delta))
                        cell = region[nearest]["cell"]
                        local = np.array(
                            [
                                feature(r["features"])
                                for r in region
                                if r["cell"] == cell
                            ]
                        )
                        f = feature(sample)
                        member = bool(
                            np.all(f >= local.min(0) - 1e-6)
                            and np.all(f <= local.max(0) + 1e-6)
                        )
                        if member and w.row["front_support"] and w.screen()[0]:
                            probe = counterfactual_recovery(c, w, assets, cfg)
                            queries.append(dict(time=float(w.d.time), **probe))
                            if not probe["success"]:
                                trace.append(dict(mode="kick", **sample))
                                if done:
                                    break
                                continue
                            q, v, t = w.d.qpos.copy(), w.d.qvel.copy(), float(w.d.time)
                            adapter = BalanceAdapter(c, w)
                            entry = t
                            m.hold = m.best_hold = 0.0
                            assert (
                                np.array_equal(q, w.d.qpos)
                                and np.array_equal(v, w.d.qvel)
                                and t == w.d.time
                            )
                    trace.append(dict(mode="kick", **sample))
                    if done:
                        break
                else:
                    target = adapter.target()
                    for _ in range(5):
                        row = w.tick(target)
                        adapter.advance_memory(row["loads"], w.m.opt.timestep)
                        sample = m.observe(row, w.m.opt.timestep)
                    trace.append(dict(mode="balance", **sample))
                    if m.failed or m.best_hold >= 5.0:
                        break
            results.append(
                dict(
                    seed=seed,
                    stochastic=stochastic,
                    handoff=entry is not None,
                    handoff_time=entry,
                    counterfactual_queries=queries,
                    takeoff=bool(w.tracker.confirmed),
                    terminal_normalized_linf_distance=float(
                        np.min(np.max(abs(points - feature(sample)), axis=1))
                    )
                    if len(points)
                    else None,
                    nearest_normalized_linf_distance=distance
                    if np.isfinite(distance)
                    else None,
                    **m.result(),
                )
            )
            csv_write(out / f"switch_{seed}_{int(stochastic)}.csv", trace)
    dump(
        out / "switching.json",
        dict(
            trials=results,
            distance_definition=dict(
                features=FEATURES, scales=SCALES.tolist(), norm="L_infinity"
            ),
            gate=(
                "Empirical envelope filter plus exact-state MuJoCo counterfactual; "
                "no live physical reset"
            ),
            region_entries=len(region),
        ),
    )
    return results


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["evaluate", "maps", "switch", "all"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--selection", type=Path)
    p.add_argument("--region", type=Path)
    p.add_argument("--roll-only", action="store_true")
    p.add_argument("--fine", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.seeds < 2:
        raise ValueError("At least two seeds required")
    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    # Always validate observation/actuation/state continuity before offline rollouts.
    subprocess.run(
        [
            __import__("sys").executable,
            "-m",
            "pytest",
            "-q",
            "experiments/recoverability/test_core.py",
        ],
        check=True,
    )
    net, hp, cfg, source = curriculum.load_source(KICK)
    inventory(a.output)
    cfg.seconds = 8.0
    dump(
        a.output / "manifest.json",
        dict(
            kick_checkpoint=str(KICK),
            kick_sha256=sha(KICK),
            config=vars(cfg),
            base_commit=source_commit(),
            source_sha256={
                str(f): sha(f) for f in Path("experiments/recoverability").glob("*.py")
            },
            arguments=vars(a),
            seed_count=a.seeds,
            ppo_updates=0,
            physics="Full MuJoCo Go2, original 1.00 morphology",
        ),
    )
    for f in Path("experiments/recoverability").glob("*.py"):
        target = a.output / "source" / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    shutil.copy2("curriculum.py", a.output / "source/curriculum.py")
    assets = Assets(a.output / "assets")
    best = (
        evaluate_candidates(
            assets, cfg, a.output, list(range(82000, 82000 + a.seeds)), a.roll_only
        )
        if a.mode in ("evaluate", "all")
        else json.loads(a.selection.read_text())["selected"]
    )
    if a.mode == "evaluate":
        return
    if best is None:
        dump(
            a.output / "decision.json",
            dict(
                reason=(
                    "No suitable frozen balance controller. Stop fusion and "
                    "prepare a separate upright experiment."
                ),
                ppo_updates=0,
            ),
        )
        return
    label = best["controller"]
    _, path, task = next(r for r in CANDIDATES if r[0] == label)
    c = SavedController(path, task)
    try:
        dump(a.output / "frozen_controller.json", c.meta)
        if a.mode in ("maps", "all"):
            cells, region = maps(
                c, assets, cfg, a.output, list(range(83000, 83000 + a.seeds)), a.fine
            )
        else:
            region = json.loads(a.region.read_text())
        if a.mode in ("switch", "all"):
            switching(
                net,
                c,
                assets,
                cfg,
                a.output,
                list(range(84000, 84000 + a.seeds)),
                region,
            )
    finally:
        c.close()


if __name__ == "__main__":
    main()
