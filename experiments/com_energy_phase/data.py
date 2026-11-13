"""Measured target provenance and event-aligned partial kick-up demonstrations."""

import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from experiments.com_energy_phase.core import measurements, Phase


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(env, directory):
    directory = Path(directory)
    bank = np.load(directory / "recoverable_states.npz")
    labels = json.loads((directory / "validated_region.json").read_text())
    candidates = []
    for i, label in enumerate(labels):
        mujoco.mj_setState(
            env.m, env.d, bank[f"{i}_integration"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(env.m, env.d)
        env.row = env.measure()
        s = measurements(env)
        f = label["features"]
        # Independent consistency checks bind archive labels to this compiled geometry.
        if abs(s["com_height"] - f["com_z"]) > 1e-7:
            raise ValueError("Reference geometry mismatch")
        if np.max(abs(env.d.qpos[env.qids] - [f[f"q{j}"] for j in range(12)])) > 1e-7:
            raise ValueError("Reference joint ordering mismatch")
        if (
            s["support_valid"]
            and f["safe"]
            and abs(env.row["pitch"] - np.pi / 2) < np.deg2rad(5)
            and not env.row["illegal_contacts"]
            and not env.row["self_contacts"]
        ):
            candidates.append((i, s, env.d.qpos.copy(), env.d.xmat[env.base].copy()))
    if not candidates:
        raise ValueError("No checked near-vertical supported reference")
    selected = min(
        candidates,
        key=lambda r: abs(
            r[1]["com_height"] - np.median([c[1]["com_height"] for c in candidates])
        ),
    )
    i, s, q, rotation = selected
    return dict(
        height=s["com_height"],
        potential=s["potential"],
        mass=s["mass"],
        rotation=rotation.tolist(),
        qpos=q.tolist(),
        sample_index=i,
        checked_candidates=len(candidates),
        checked_indices=[c[0] for c in candidates],
        measurements=s,
        source=str(directory),
        bank_sha256=sha(directory / "recoverable_states.npz"),
        labels_sha256=sha(directory / "validated_region.json"),
        validation=(
            "Archived successful rollouts in validated balance cells; "
            "same-geometry COM and joints rechecked. "
            "No transition or broad basin claim."
        ),
    )


def align(data, mass):
    """Align only observed physical events; never synthesize handstand labels."""
    events = []
    encodings = np.zeros((len(data["time"]), 3), np.float32)
    weights = np.zeros(len(encodings), np.float32)
    for scale in np.unique(data["scale"]):
        for seed in np.unique(data["seed"]):
            ids = np.flatnonzero((data["scale"] == scale) & (data["seed"] == seed))
            if not len(ids):
                continue
            ids = ids[np.argsort(data["time"][ids])]
            times = data["time"][ids]
            loads = data["contacts"][ids]
            front = loads[:, :2].sum(1)
            rear = loads[:, 2:].sum(1)
            loaded = np.flatnonzero((loads[:, 2:] > 0.08 * mass * 9.81).all(1))
            if not len(loaded):
                continue
            loading = np.flatnonzero(
                (front > rear) & (np.arange(len(ids)) >= loaded[0])
            )
            release = np.flatnonzero(
                (loads[:, 2:] < 0.02 * mass * 9.81).all(1)
                & (np.arange(len(ids)) > loaded[0])
            )
            if not len(release):
                continue
            lift = int(release[0])
            load = int(loading[0]) if len(loading) else int(loaded[0])
            # Peak rear knee extension (larger calf angle) before observed release.
            extension = int(
                np.argmax(data["qpos"][ids[: lift + 1]][:, [15, 18]].mean(1))
            )
            # Strict increasing knots preserve ordering where observations coincide.
            knots = np.maximum.accumulate(
                [max(0.02, times[load]), max(0.04, times[extension]), times[lift]]
            )
            for j in range(1, 3):
                knots[j] = max(knots[j], knots[j - 1] + 0.02)
            phase = Phase(tuple(knots) + (knots[2] + 0.2, knots[2] + 0.6), True)
            for local, index in enumerate(ids):
                phase.update(times[local], local >= loaded[0], local >= lift, 0, False)
                encodings[index] = phase.encode()
                weights[index] = 1.0 if local <= lift else 0.1
            events.append(
                dict(
                    scale=float(scale),
                    seed=int(seed),
                    loading=float(times[load]),
                    extension=float(times[extension]),
                    liftoff=float(times[lift]),
                    early_rotation=float(times[-1]),
                    knots=knots.tolist(),
                )
            )
    if not events:
        raise ValueError("No loaded rear release events in demonstrations")
    full = [e for e in events if e["scale"] == 1.0]
    knots = np.median([e["knots"] for e in full], axis=0)
    return (
        encodings,
        weights,
        events,
        tuple(knots) + (float(knots[2] + 0.2), float(knots[2] + 0.6)),
    )
