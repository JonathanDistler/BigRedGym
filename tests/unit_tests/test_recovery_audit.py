"""Audit interfaces, trajectory counts, asset integrity and evaluation cohorts."""

import json
from pathlib import Path

import pytest

from scripts.audit_recovery_dataset import audit_bank, audit_interface
from scripts.evaluate_kickup_recovery import discover_checkpoints, summarize_pair
from scripts.verify_recovery_assets import checksum, verify_manifest

DATA = Path("datasets/go2kickup_recovery_train_v2")


@pytest.fixture(scope="module")
def interface_report():
    if not DATA.exists():
        pytest.skip("Physical recovery dataset is required")
    return audit_interface(DATA)


def test_full_observation_mapping_and_checkpoint_normalization(interface_report):
    actor = interface_report["observations"]["actor"]
    critic = interface_report["observations"]["critic"]
    actor_expected = [
        ("base_ang_vel", 0, 3),
        ("projected_gravity", 3, 6),
        ("commands", 6, 10),
        ("dof_pos_obs", 10, 22),
        ("dof_vel", 22, 34),
        ("dof_pos_target", 34, 46),
        ("handstand_stage_obs", 46, 47),
        ("cpg_phase_obs", 47, 55),
        ("cpg_contact_obs", 55, 59),
    ]
    critic_expected = [
        ("base_height", 0, 1),
        ("base_lin_vel", 1, 4),
        ("base_ang_vel", 4, 7),
        ("projected_gravity", 7, 10),
        ("commands", 10, 14),
        ("dof_pos_obs", 14, 26),
        ("dof_vel", 26, 38),
        ("dof_pos_target", 38, 50),
        ("handstand_stage_obs", 50, 51),
        ("cpg_phase_obs", 51, 59),
        ("cpg_contact_obs", 59, 63),
    ]
    for mapping, expected in ((actor, actor_expected), (critic, critic_expected)):
        assert [
            (x["name"], x["start"], x["stop"]) for x in mapping["slices"]
        ] == expected
    assert actor["dimension"] == 59
    assert critic["dimension"] == 63
    assert interface_report["normalization"]["actor"] == dict(enabled=False, buffers={})
    norm = interface_report["normalization"]["critic"]
    assert norm["enabled"]
    assert norm["buffers"]["obs_rms.running_mean"]["shape"] == [63]
    assert norm["buffers"]["obs_rms.running_var"]["shape"] == [63]
    assert norm["buffers"]["obs_rms.count"]["min"] > 1
    assert interface_report["normalization_unchanged_after_eval"]
    assert interface_report["maximum_original_observation_difference"] == 0
    assert interface_report["maximum_original_action_difference"] == 0


def test_all_native_cpg_restores_and_procedural_stages(interface_report):
    assert interface_report["all_replay_states_checked"] == 98
    assert interface_report["maximum_phase_observation_error"] == 0
    assert interface_report["cpg_config"]["modes"] == [
        "stance",
        "stance",
        "swing",
        "swing",
    ]
    stages = interface_report["procedural_reset_coverage"]
    assert [stages[str(i)]["resets"] for i in range(3)] == [32, 32, 32]
    assert stages["0"]["observed_ranges"]["com_speed"][1] < 1e-6
    assert stages["1"]["observed_ranges"]["com_speed"][1] > 0
    assert (
        0
        < stages["2"]["observed_ranges"]["max_horizontal_base_speed_component"][1]
        <= 0.25
    )


def test_real_stage_coverage_counts_trajectories():
    if not DATA.exists():
        pytest.skip("Physical recovery dataset is required")
    report = audit_bank(DATA)
    assert (report["snapshots"], report["trajectories"]) == (98, 13)
    assert [
        (report["stages"][s]["snapshots"], report["stages"][s]["trajectories"])
        for s in ("3", "4", "5")
    ] == [(29, 5), (46, 3), (98, 13)]
    assert report["stage_overlap_snapshots"]["3_4"] == 23
    assert report["stages"]["4"]["largest_trajectory_share"] == pytest.approx(19 / 46)
    assert report["trajectory_outcomes"].get("recovered", 0) == 0
    assert report["missing_trajectory_seeds"] == [10010, 10011, 10012]


def test_checkpoint_discovery_is_numeric_and_includes_initial(tmp_path):
    for name in ("model_100.pt", "model_0.pt", "model_25.pt", "model_50.pt.tmp"):
        (tmp_path / name).touch()
    assert [p.name for p in discover_checkpoints(tmp_path)] == [
        "model_0.pt",
        "model_25.pt",
        "model_100.pt",
    ]
    with pytest.raises(ValueError, match="No completed"):
        discover_checkpoints(tmp_path / "missing")


def test_paired_summary_aligns_by_seed_and_rejects_missing_trials():
    baseline = dict(
        successes=1,
        trials=[
            dict(seed=6000, success=True, hold_s=5.0),
            dict(seed=6001, success=False, hold_s=1.0),
        ],
    )
    candidate = dict(
        successes=1,
        trials=[
            dict(seed=6001, success=True, hold_s=5.1),
            dict(seed=6000, success=False, hold_s=0.5),
        ],
    )
    report = summarize_pair(baseline, candidate)
    assert report["wins"] == report["losses"] == 1
    assert report["success_rate_difference"] == 0
    assert report["mean_hold_difference_s"] == pytest.approx(-0.2)
    candidate["trials"].pop()
    with pytest.raises(ValueError, match="seed mismatch"):
        summarize_pair(baseline, candidate)


def test_preflight_detects_missing_and_tampered_assets(tmp_path):
    import subprocess

    subprocess.run(
        ["git", "init", "--initial-branch=ppo_kickup", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    asset = tmp_path / "train.npz"
    asset.write_bytes(b"original physical states")
    manifest = dict(
        branch="ppo_kickup",
        files={"train.npz": dict(category="training", sha256=checksum(asset))},
    )
    assert verify_manifest(tmp_path, manifest)["verified"]
    asset.write_bytes(b"tampered")
    assert not verify_manifest(tmp_path, manifest)["verified"]
    manifest["files"]["missing.pt"] = dict(category="training", sha256="missing")
    report = verify_manifest(tmp_path, manifest)
    assert not report["files"]["missing.pt"]["exists"]
    assert not report["verified"]


def test_dataset_hash_must_match_provenance(tmp_path):
    import shutil

    if not DATA.exists():
        pytest.skip("Physical recovery dataset is required")
    for name in ("train.npz", "labels.json", "provenance.json"):
        shutil.copy2(DATA / name, tmp_path / name)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance["dataset_sha256"] = "tampered"
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="SHA256"):
        audit_bank(tmp_path)
