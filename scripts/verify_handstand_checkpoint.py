"""Compare trusted complete checkpoints after uninterrupted vs restarted simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def assert_equal(left, right, path="checkpoint"):
    if isinstance(left, torch.Tensor):
        if not torch.isfinite(left).all() or not torch.equal(left, right):
            raise ValueError(f"Tensor mismatch/nonfinite: {path}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise ValueError(f"Array mismatch: {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Keys mismatch: {path}")
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError(f"Length mismatch: {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{i}]")
    elif left != right:
        raise ValueError(f"Value mismatch: {path}")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    left = torch.load(args.left, weights_only=False, map_location="cpu")
    right = torch.load(args.right, weights_only=False, map_location="cpu")
    for key in [
        "kind",
        "schema",
        "policy",
        "optimizer",
        "update",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "reference_sha256",
        "curriculum",
    ]:
        assert_equal(left[key], right[key], key)
    if left.get("best_payload") is not None and right.get("best_payload") is not None:
        for key in [
            "policy",
            "optimizer",
            "update",
            "torch_rng",
            "numpy_rng",
            "python_rng",
            "curriculum",
        ]:
            assert_equal(
                left["best_payload"][key],
                right["best_payload"][key],
                "selected_best." + key,
            )
    result = dict(
        exact_resume=True,
        update=left["update"],
        checked="policy, optimizer, RNGs, normalization buffers, curriculum, reference",
        scope=(
            "Real simulator PPO; runtime metrics and output paths "
            "intentionally excluded"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
