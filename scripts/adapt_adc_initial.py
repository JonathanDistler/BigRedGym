"""Adapt a trusted shared BC actor to the common 73/103 architecture.

Shared actor columns are copied exactly; new columns are zero. The explicitly
chosen reference contributes config and critic weights, never a second controller.
"""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.data import network, sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("bc_actor", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.output.exists():
        raise FileExistsError(a.output)
    net, parent = network(a.reference)
    saved = torch.load(a.bc_actor, weights_only=False, map_location="cpu")
    actor = copy.deepcopy(saved["actor"])
    weight = actor["0.weight"]
    if weight.shape[1] > 73:
        raise ValueError("Incompatible input ordering/dimensions")
    actor["0.weight"] = torch.cat(
        [weight, torch.zeros((len(weight), 73 - weight.shape[1]))], 1
    )
    net.actor.load_state_dict(actor)
    parent["network"] = net.state_dict()
    parent["initialization_provenance"] = dict(
        actor=str(a.bc_actor),
        actor_sha256=sha(a.bc_actor),
        critic_and_config=str(a.reference),
        reference_sha256=sha(a.reference),
        new_actor_columns="zero",
    )
    parent.pop("optimizer", None)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        torch.save(parent, f)
    print(a.output)


if __name__ == "__main__":
    main()
