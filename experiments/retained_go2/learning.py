"""Adaptive failure sampling, immutable skill bests, and catastrophic-regression stop."""

import json, shutil
from pathlib import Path
import numpy as np
from experiments.unified_go2.core import Vector as SerialVector, save_json, digest
from experiments.front_catch.io import atomic_checkpoint
from experiments.retained_go2.evaluation import evaluate as suite, retention

OUTPUT = None
CERTIFICATE = None
BASELINE = None
RETENTION_HASH = None


class Vector(SerialVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.competence = np.array([1.0, 0.0, 0.75])
        self.trials = np.zeros(3, dtype=int)

    def probabilities(self):
        floor = np.array([0.25, 0.20, 0.25])
        if self.stage >= 2:
            floor = np.array([0.35, 0.20, 0.25])
        if self.stage >= 3:
            floor = np.array([0.50, 0.15, 0.25])
        if self.competence[2] < 0.5:
            floor = np.array([0.25, 0.20, 0.55])
        failure = np.maximum(1 - self.competence, 0.02)
        return floor + (1 - floor.sum()) * failure / failure.sum()

    def start(self, i, kind=None):
        if kind is None:
            kind = self.rng.choice(
                ["standing", "bridge", "balance"], p=self.probabilities()
            )
        return super().start(i, kind)

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        for row in results:
            episode = row[5]
            if episode:
                k = ["standing", "bridge", "balance"].index(
                    episode["start_distribution"]
                )
                success = (
                    episode["certified_standing_takeoff"]
                    if k == 0
                    else episode["handstand_seconds"] >= 5
                )
                self.competence[k] = 0.9 * self.competence[k] + 0.1 * float(success)
                self.trials[k] += 1
        return results

    def state(self):
        state = super().state()
        state.update(
            competence=self.competence.copy(),
            trials=self.trials.copy(),
            retention_contract=RETENTION_HASH,
            skill_bests=json.loads((OUTPUT / "skill_bests.json").read_text())
            if OUTPUT is not None and (OUTPUT / "skill_bests.json").exists()
            else None,
        )
        return state

    def restore(self, state):
        if state.get("retention_contract") != RETENTION_HASH:
            raise ValueError("Retention evaluation contract changed")
        if OUTPUT is not None and state.get("skill_bests") is not None:
            path = OUTPUT / "skill_bests.json"
            if path.exists() and json.loads(path.read_text()) != state["skill_bests"]:
                save_json(
                    OUTPUT
                    / f"selection_before_restore_{len(list(OUTPUT.glob('selection_before_restore_*'))):04}.json",
                    json.loads(path.read_text()),
                )
            save_json(path, state["skill_bests"])
        result = super().restore(state)
        self.competence = state["competence"].copy()
        self.trials = state["trials"].copy()
        return result


def configure(a, certificate):
    global OUTPUT, CERTIFICATE, BASELINE, RETENTION_HASH
    OUTPUT = a.output
    CERTIFICATE = certificate
    baseline_path = Path(certificate["baseline_file"])
    BASELINE = json.loads(baseline_path.read_text())["adc"]
    import hashlib

    RETENTION_HASH = hashlib.sha256(
        (json.dumps(certificate, sort_keys=True) + digest(baseline_path)).encode()
    ).hexdigest()
    if not a.resume:
        if digest(a.initial) != certificate["initial_sha"]:
            raise ValueError(
                "Initial actor differs from validated retention checkpoint"
            )
        shutil.copy2(a.initial, a.output / "initial_verified.pt")
        save_json(
            a.output / "skill_bests.json",
            dict(
                kick=dict(
                    checkpoint="initial_verified.pt",
                    validated=True,
                    score=[
                        1.0,
                        float(np.deg2rad(certificate["mean_standing_peak_degrees"])),
                    ],
                ),
                balance=dict(
                    checkpoint="initial_verified.pt",
                    validated=True,
                    score=[
                        certificate["balance_successes"] / certificate["balance_n"],
                        float(np.mean(certificate["balance_holds"])),
                    ],
                ),
                combined=dict(
                    checkpoint="initial_verified.pt",
                    validated=True,
                    score=[
                        1.0,
                        0.0,
                        1.0,
                        certificate["balance_successes"] / certificate["balance_n"],
                        0.0,
                    ],
                ),
                bridge=None,
            ),
        )


def evaluate(*args, **kwargs):
    rows = suite(*args, **kwargs)
    if OUTPUT is None:
        return rows
    result = retention(rows, BASELINE)
    iteration = (
        max(int(p.stem.split("_")[-1]) for p in OUTPUT.glob("checkpoint_*.pt")) + 1
    )
    save_json(OUTPUT / f"retention_{iteration:06}.json", result)
    groups = {
        k: [r for r in rows if r["distribution"] == k]
        for k in ["standing", "balance", "bridge"]
    }
    avg = lambda group, key: float(np.mean([r[key] for r in groups[group]]))
    scores = dict(
        kick=[
            avg("standing", "certified_standing_takeoff"),
            avg("standing", "peak_collision_free_pitch"),
        ],
        balance=[avg("balance", "success"), avg("balance", "handstand_seconds")],
        bridge=[
            avg("bridge", "success"),
            avg("bridge", "handstand_seconds"),
            avg("bridge", "peak_collision_free_pitch"),
        ],
        combined=[
            float(result["passed"]),
            avg("standing", "standing_success"),
            avg("standing", "certified_standing_takeoff"),
            avg("balance", "success"),
            avg("bridge", "handstand_seconds"),
        ],
    )
    index = json.loads((OUTPUT / "skill_bests.json").read_text())
    net = args[0]
    for skill, score in scores.items():
        previous = index.get(skill)
        # Initial verified endpoint reference is never overwritten by a regression.
        if skill in ["kick", "balance", "combined"] and not result["passed"]:
            continue
        if (
            previous
            and previous.get("score") is not None
            and score <= previous["score"]
        ):
            continue
        name = f"best_{skill}_{iteration:06}.pt"
        atomic_checkpoint(
            OUTPUT / name,
            dict(
                network=net.state_dict(),
                score=score,
                skill=skill,
                iteration=iteration,
                retention=result,
            ),
        )
        index[skill] = dict(
            checkpoint=name, score=score, validated=bool(result["passed"])
        )
    save_json(OUTPUT / "skill_bests.json", index)
    if not result["passed"]:
        (OUTPUT / "STOP").write_text(
            "Endpoint regression: immutable current boundary and initial_verified.pt preserved. Inspect retention JSON before any continuation.\n"
        )
    return rows


def promote(stage, rows, reference):
    stand = [r for r in rows if r["distribution"] == "standing"]
    bal = [r for r in rows if r["distribution"] == "balance"]
    bridge = [r for r in rows if r["distribution"] == "bridge"]
    if (
        not all(r["certified_standing_takeoff"] for r in stand)
        or sum(r["success"] for r in bal) < 3
    ):
        return 0
    hold = max((r["handstand_seconds"] for r in bridge), default=0)
    if stage == 0:
        return 1
    if stage == 1 and hold >= 1:
        return 2
    if stage == 2 and hold >= 5:
        return 3
    return stage
