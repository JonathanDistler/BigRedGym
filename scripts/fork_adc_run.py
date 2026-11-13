"""Explicit exact-boundary fork of a selected checkpoint; no policy-only reset."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.data import sha
from scripts.run_adc_front_capture import sources


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    if saved["protocol"]["source_sha256"] != sources():
        raise ValueError(
            "Source mismatch: use archived source or explicit evaluator migration"
        )
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.checkpoint, a.output / a.checkpoint.name)
    (a.output / "protocol.json").write_text(json.dumps(saved["protocol"], indent=2))
    (a.output / "fork.json").write_text(
        json.dumps(
            dict(
                parent=str(a.checkpoint),
                parent_sha256=sha(a.checkpoint),
                iteration=saved["iteration"],
                all_training_state_preserved=True,
            ),
            indent=2,
        )
    )
    print("Exact checkpoint copy:", a.output / a.checkpoint.name)


if __name__ == "__main__":
    main()
