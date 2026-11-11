"""Package audited residual code, original dependencies and completed evidence."""

import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_recovery_assets import build_manifest, checksum

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    a = parser.parse_args()
    if a.manifest.exists() or a.bundle.exists():
        raise FileExistsError("Use new manifest and bundle paths")
    expected = json.loads(
        (ROOT / "output/recovery_frontier_protected_after_resume.json").read_text()
    )
    for path, entry in expected.items():
        if checksum(ROOT / path) != entry["sha256"]:
            raise ValueError(f"Protected asset changed: {path}")
    manifest = build_manifest(ROOT, "datasets/recovery_frontier_v4")
    extra = [
        ROOT / "uv.lock",
        ROOT / "README.md",
        ROOT / "datasets/recovery_frontier_v3/train.npz",
        ROOT / "datasets/recovery_frontier_assets_manifest_v2.json",
        ROOT / "output/recovery_frontier_protected_after_resume.json",
        ROOT / "output/recovery_frontier_learning_resumed_v1/summary.json",
    ]
    for checkpoint in (
        manifest["checkpoint"],
        "logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt",
    ):
        effective = (ROOT / checkpoint).parent / "effective_config.json"
        if effective.is_file():
            extra.append(effective)
    extra += list((ROOT / "tests").rglob("*.py"))
    extra += list((ROOT / "docs").glob("go2kickup*.md"))
    for pattern in (
        "frozen_residual_zero_v2/*",
        "frozen_residual_commands_v1.json",
        "frozen_residual_comparison_v1.json",
        "frozen_residual_comparison_v2.json",
        "frozen_residual_authority_v1.json",
        "frozen_residual_protected_code_v1.json",
        "frozen_residual_episode_rewards_v1.json",
        "frozen_residual_standing_v1/*",
        "balance_early_replay_v1/*/*",
        "balance_stage_zero_rewards_v1.json",
        "balance_action_comparison_v2.json",
        "balance_first_updates_v2/audit.json",
        "balance_critic_control_v1/audit.json",
    ):
        extra += list((ROOT / "output").glob(pattern))
    extra += list((ROOT / a.run_dir).rglob("*"))
    for path in extra:
        if path.is_file():
            name = str(path.relative_to(ROOT))
            manifest["files"][name] = dict(
                category="evidence", bytes=path.stat().st_size, sha256=checksum(path)
            )
    with a.manifest.open("x") as f:
        json.dump(manifest, f, indent=2)
    with tarfile.open(a.bundle, "x:gz") as archive:
        for name in sorted(manifest["files"]):
            archive.add(ROOT / name, arcname=name, recursive=False)
        archive.add(a.manifest, arcname=str(a.manifest), recursive=False)
    with Path(str(a.bundle) + ".sha256").open("x") as f:
        f.write(f"{checksum(a.bundle)}  {a.bundle}\n")
    print(f"{len(manifest['files'])} files: {a.bundle}")


if __name__ == "__main__":
    main()
