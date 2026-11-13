"""Deployment contracts; tests don't alter PPO or policy reward implementations."""

import ast
import hashlib
import json
import zipfile

import pytest
import torch
from deployment.molab import common, workflow


def test_matched_budgets_for_all_counts():
    base = common.load(common.ROOT / "deployment/molab/experiment.json")
    values = []
    for n in (8, 16, 32, 64):
        cfg = workflow.effective_config(dict(base, num_envs=n))
        values.append(
            cfg["num_envs"] * cfg["horizon"] * cfg["updates"] * len(cfg["seeds"]) * 3
        )
    assert values == [30720] * 4


def test_budget_and_config_rejection():
    base = common.load(common.ROOT / "deployment/molab/experiment.json")
    for change in (dict(num_envs=7), dict(updates=1000), dict(checkpoint_every=0)):
        with pytest.raises(ValueError):
            workflow.effective_config(dict(base, **change))


def test_cells_are_kernel_version_independent_and_separate():
    names = []
    for name in ("MOLAB_START.py", "MOLAB_TRAIN.py", "MOLAB_BACKUP.py"):
        tree = ast.parse((common.ROOT / name).read_text())
        assert all(isinstance(n, (ast.FunctionDef, ast.Expr)) for n in tree.body)
        names.extend(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
        assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    assert len(names) == len(set(names)) == 3


def test_corrupt_checkpoint_is_preserved_and_tmp_ignored(tmp_path):
    valid = dict(
        iteration=0,
        protocol={},
        history={},
        torch_rng=torch.get_rng_state(),
        numpy_rng=None,
        runs={
            "A": dict(
                network={},
                optimizer={},
                torch_rng=None,
                worlds=[],
                actors=[],
                critics=[],
            )
        },
    )
    torch.save(valid, tmp_path / "checkpoint_000000.pt")
    bad = tmp_path / "checkpoint_000001.pt"
    bad.write_bytes(b"partial checkpoint")
    (tmp_path / ".checkpoint_pending.tmp").write_bytes(b"pending")
    result = workflow.valid_checkpoints(tmp_path)
    assert len(result) == 1 and result[0][1] == 0 and not bad.exists()
    quarantined = list((tmp_path / "quarantine").glob("*.bad"))
    assert (
        len(quarantined) == 1 and quarantined[0].read_bytes() == b"partial checkpoint"
    )
    assert (tmp_path / ".checkpoint_pending.tmp").exists()


def test_atomic_json_replaces_only_metadata(tmp_path):
    target = tmp_path / "state.json"
    common.atomic_json(target, {"iteration": 1})
    common.atomic_json(target, {"iteration": 2})
    assert common.load(target) == {"iteration": 2}


def test_manifest_all_required_files():
    manifest = common.verify_manifest()
    for name in [
        "scripts/run_front_catch.py",
        "resources/robots/go2/urdf/go2.urdf",
        "output/front_catch_preparation_20261112_v3/reset_bank.pt",
        "MOLAB_START.py",
        "deployment/molab/requirements.lock",
    ]:
        assert name in manifest["files"]
    assert (
        manifest["initial_sha256"]
        == "f20e26503b24bd5ed60ae43d983300f0cbf16d757ba2831ebce5a5ca9e22045c"
    )


def test_exports_are_immutable_and_checksummed(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    run_dir = root / "output/run"
    run_dir.mkdir(parents=True)
    (root / ".molab").mkdir()
    source = root / "source.py"
    source.write_text("immutable source\n")
    (run_dir / "metrics.csv").write_text("step,reward\n1,0\n")
    (run_dir / "checkpoint_000001.pt").write_bytes(b"immutable checkpoint")
    (run_dir / "unfinished.tmp").write_text("partial")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                files={"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                source_revision="test",
            )
        )
    )
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "MANIFEST", manifest)
    monkeypatch.setattr(workflow, "ROOT", root)
    monkeypatch.setattr(workflow, "MANIFEST", manifest)
    first = workflow.export(run_dir, tmp_path / "exports")
    original = first.read_bytes()
    second = workflow.export(run_dir, tmp_path / "exports")
    assert first != second and first.read_bytes() == original
    with zipfile.ZipFile(first) as archive:
        assert not any(n.endswith(".tmp") for n in archive.namelist())
        hashes = json.loads(archive.read("EXPORT_MANIFEST.json"))["files"]
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


def test_eight_training_worlds_use_separate_evaluation_without_ppo(
    tmp_path, monkeypatch
):
    import copy
    import numpy as np
    import scripts.run_front_catch as runner
    from deployment.molab import entrypoint
    from experiments.front_catch.core import CatchConfig
    from experiments.morphology.learning.environment import Assets
    from experiments.front_catch.vector import KINDS

    bank = torch.load(
        common.ROOT / "output/front_catch_preparation_20261112_v3/reset_bank.pt",
        weights_only=False,
        map_location="cpu",
    )
    assets = Assets(tmp_path / "assets")
    cfg = CatchConfig(catch_mode="C", comparison_suite="anticipatory_v1")
    original_vector, original_evaluate = runner.Vector, runner.evaluate

    def probe():
        with runner.Vector(
            assets,
            {"C": cfg, "evaluation": copy.deepcopy(cfg)},
            bank,
            8,
            seed=42,
            workers=0,
        ) as vector:
            actor, _ = vector.reset("C", [1.0, 0, 0, 0, 0, 0])
            assert vector.n == 8 and actor.shape == (8, 73)
            view = vector.separate_evaluation
            assert view.n == 16
            specs = [dict(kind="standing", seed=96000 + i) for i in range(4)]
            specs += [
                dict(kind=kind, index=0, seed=97000 + i)
                for i, kind in enumerate(KINDS[1:])
            ]
            specs += [None] * (16 - len(specs))
            eval_actor, _ = view.evaluation_reset("evaluation", "C", specs)
            assert eval_actor.shape == (16, 73)
            active = np.array([spec is not None for spec in specs])
            result = view.evaluation_step("evaluation", np.zeros((16, 12)), active)
            assert len(result) == 16 and active.sum() == 9

    monkeypatch.setattr(runner, "main", probe)
    try:
        entrypoint.main()
    finally:
        runner.Vector, runner.evaluate = original_vector, original_evaluate


def test_existing_pins_skip_network_resolution(tmp_path):
    import sys
    from deployment.molab.bootstrap import packages_match

    assert packages_match(
        sys.executable, common.ROOT / "deployment/molab/requirements.lock"
    )
    wrong = tmp_path / "wrong.lock"
    wrong.write_text("numpy==0.0.0\n")
    assert not packages_match(sys.executable, wrong)
