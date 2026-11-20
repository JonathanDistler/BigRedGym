"""Independent software-only replay/export gate."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.trajectory_optimization.validation import replay, export

if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--export", type=Path)
    args = p.parse_args()
    if args.export:
        print(json.dumps(export(args.candidate, args.export), indent=2))
    else:
        print(json.dumps(replay(args.candidate), indent=2))
