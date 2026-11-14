"""Replay external Go2 handstand commands; no live policy or hardware control."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym.envs.go2.handstand_replay import Replay, build_model, load_trajectory


def main():
    """Print measured strict success and exit nonzero on failure or divergence."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="External schema-1 NPZ with standing state and recorded commands",
    )
    parser.add_argument(
        "--trajectory-sha256",
        required=True,
        help="Expected SHA256 from the artifact provider",
    )
    args = parser.parse_args()
    try:
        model = build_model()
        trajectory = load_trajectory(args.trajectory, args.trajectory_sha256, model)
        result = Replay(model, trajectory).run()
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Handstand replay: {exc}\n")
    print(json.dumps(result, indent=2))
    return (
        0
        if (
            result["five_second_success"]
            and result["replay_matches"]
            and result["prefix_safe"]
            and result["certified_takeoff"]
            and result["front_capture"]
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
