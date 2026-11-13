"""Copy immutable preparation, revalidate actual states and short native rollouts."""

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from experiments.front_catch.data import sha, config, validate_state
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.morphology.learning.environment import Assets
from scripts.run_adc_front_capture import initial
from scripts.run_com_energy_phase import dump
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.source / "manifest.json").read_text())
    bank = torch.load(
        a.source / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    net, parent = initial(manifest["initial"])
    cfg = config(parent, "B")
    cfg.adc_enabled = True
    env = CaptureEnvironment(Assets(a.output / "assets"), cfg)
    rows = []
    new = {split: {kind: [] for kind in groups} for split, groups in bank.items()}
    for split, groups in bank.items():
        for kind, items in groups.items():
            for i, item in enumerate(items):
                try:
                    obs, _ = env.restore_start(item["state"], kind)
                    valid, details = validate_state(env)
                    applied = np.max(abs(env.d.qfrc_applied[env.vids]) / env.effort)
                    initial_valid = valid and applied <= 1.000001
                    minimum = 1.0
                    done = False
                    for _ in range(5):
                        with torch.no_grad():
                            action = net.actor(torch.tensor(obs)).tanh().numpy()
                        (obs, _), _, done, info = env.step(action)
                        valid, _ = validate_state(env)
                        minimum = min(minimum, float(valid))
                        if done:
                            break
                    admitted = initial_valid and minimum == 1 and not done
                    reason = (
                        "native 100ms rollout"
                        if admitted
                        else "state/torque/short-rollout failure"
                    )
                    rows.append(
                        dict(
                            split=split,
                            kind=kind,
                            index=i,
                            admitted=admitted,
                            initial_torque_ratio=float(applied),
                            reason=reason,
                            **details,
                        )
                    )
                    if admitted:
                        new[split][kind].append(item)
                except ValueError as exc:
                    rows.append(
                        dict(
                            split=split,
                            kind=kind,
                            index=i,
                            admitted=False,
                            reason=str(exc),
                        )
                    )
    for split, groups in new.items():
        for kind, items in groups.items():
            if not items:
                raise RuntimeError(
                    f"No dynamically checked {split}/{kind}; do not fabricate states"
                )
    torch.save(new, a.output / "reset_bank.pt")
    shutil.copyfile(
        a.source / "bc_demonstrations.npz", a.output / "bc_demonstrations.npz"
    )
    manifest.update(
        bank_sha256=sha(a.output / "reset_bank.pt"),
        dataset_sha256=sha(a.output / "bc_demonstrations.npz"),
        parent_manifest_sha256=sha(a.source / "manifest.json"),
        bank_counts={
            split: {kind: len(v) for kind, v in groups.items()}
            for split, groups in new.items()
        },
        validation=(
            "Original geometry/limits, torque envelope and 100ms actor "
            "rollout; no stability guarantee"
        ),
    )
    dump(a.output / "manifest.json", manifest)
    save_csv(a.output / "reset_validation.csv", rows)
    print(manifest["bank_counts"])


if __name__ == "__main__":
    main()
