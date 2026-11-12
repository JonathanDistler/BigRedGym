"""Bounded action perturbation assay; never demonstrations or trained-policy claims."""

import json
from pathlib import Path
import torch
import numpy as np
import curriculum
from experiments.handstand_v2.environment import V2Environment
from experiments.handstand_v2.run import ORIGINAL
from experiments.morphology.learning.environment import Assets


def main():
    out = Path("output/handstand_v2_authority_probe_20261111")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    net, hp, cfg, _ = curriculum.load_source(ORIGINAL)
    assets = Assets(out / "assets")
    rows = []
    candidates = [
        ("original", 0.0, 0.0),
        ("front_flex", -0.02, 0.0),
        ("front_extend", 0.02, 0.0),
        ("rear_retract", 0.0, -0.03),
        ("combined", -0.02, -0.03),
    ]
    for label, front, rear in candidates:
        for seed in range(70400, 70403):
            env = V2Environment(assets, cfg)
            a, _ = env.reset(1.0, seed)
            while True:
                with torch.no_grad():
                    action = net.actor(torch.tensor(a)).tanh().numpy()
                if 0.65 <= env.d.time <= 0.95:
                    for index, name in enumerate(env.names):
                        if name.endswith("calf_joint"):
                            action[index] += (
                                front if name.startswith(("FL", "FR")) else rear
                            )
                (a, _), _, done, info = env.step(np.clip(action, -1, 1))
                if done:
                    break
            m = env.measurement
            rows.append(
                dict(
                    candidate=label,
                    seed=seed,
                    front_calf_action_bias=front,
                    rear_calf_action_bias=rear,
                    takeoff=bool(info["takeoff"]),
                    loaded_pitch=float(np.degrees(m.loaded_peak)),
                    certified_pitch=float(np.degrees(m.certified_peak)),
                    slip=m.slip,
                    recovered=m.recovered,
                    first_invalid=m.first_invalid,
                )
            )
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
