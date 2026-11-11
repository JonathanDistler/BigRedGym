"""Measured reset curriculum and native recovery objective contracts."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2kickup_recovery import Go2KickupRecovery
from gym.envs.go2.recovery_frontier import (
    install_samples,
    valid_recovery,
    sample_from_arrays,
)
from gym.envs.go2.recovery_state import RecoveryBank
from gym.utils.task_registry import task_registry
from scripts.evaluate_kickup_handoff import Policy
from scripts.reproduce_transition_hold import BALANCE
from scripts.train_kickup_recovery import frontier_configuration

ROOT = Path("datasets/recovery_frontier_v4")


@pytest.fixture
def bank():
    if not (ROOT / "train.npz").exists():
        pytest.skip("Build expanded measured bank first")
    return RecoveryBank(ROOT / "train.npz")


def test_recoverable_start_and_rollout_family_splits(bank):
    assert bank.version == 2
    assert bank.data["frozen_success"][bank.indices(0)].all()
    assert len(np.unique(bank.data["source_id"])) >= 40
    sources, families = [], []
    for split in ("train", "validation", "heldout"):
        d = RecoveryBank(ROOT / f"{split}.npz").data
        sources.append(set(d["source_id"]))
        families.append(set(d["family"]))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not sources[i] & sources[j]
        assert not families[i] & families[j]
    assert not any("6003" in s for s in sources[1] | sources[2])


def test_mixture_preserves_recoveries_and_samples_sources_uniformly(bank):
    torch.manual_seed(12)
    counts = Counter()
    sources = Counter()
    for _ in range(2000):
        bank.sample(5)
        i = bank.last_sample_index
        counts[int(bank.data["stratum"][i])] += 1
        if bank.data["stratum"][i] == 0:
            sources[str(bank.data["source_id"][i])] += 1
    assert counts[0] / 2000 == pytest.approx(0.4, abs=0.04)
    assert counts[1] / 2000 == pytest.approx(0.4, abs=0.04)
    assert counts[2] / 2000 == pytest.approx(0.2, abs=0.04)
    assert len(sources) > 4
    for _ in range(30):
        bank.sample(0)
        assert bank.data["frozen_success"][bank.last_sample_index]


@torch.inference_mode()
def test_native_frontier_reset_and_checkpoint_compatibility(bank):
    cfg, train = frontier_configuration(ROOT / "train.npz")
    cfg.env.num_envs = 2
    cfg.seed = 7
    train.runner.device = "cpu"
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    backend = MuJocoCPUBackend()
    env = Go2KickupRecovery(cfg, "cpu", True, backend)
    try:
        runner = task_registry.make_alg_runner(env, train)
        saved = torch.load(BALANCE, weights_only=True, map_location="cpu")
        runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        runner.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        runner.switch_to_eval()
        assert runner.get_obs(runner.actor_cfg["obs"]).shape == (2, 59)
        assert runner.get_obs(runner.critic_cfg["obs"]).shape == (2, 63)
        env.cfg.handstand_curriculum.replay_fraction = 0
        for stage in range(6):
            env.curriculum_level = stage
            env._reset_idx(torch.ones(2, dtype=torch.bool))
            for _ in range(3):
                runner.set_actions(
                    runner.actor_cfg["actions"], runner.get_inference_actions(), False
                )
                env.step()
            assert torch.isfinite(env.root_states).all()
            for term in (
                "recovery_com_offset",
                "recovery_com_velocity",
                "recovery_hold",
            ):
                assert torch.isfinite(getattr(env, "_reward_" + term)()).all()
        assert env.recovery_sampling_counts
        assert env.training_state_dict()["recovery_sampling_counts"]
        # No fixed-iteration gate: even maximal partial holds cannot promote.
        env.curriculum_level = 0
        env._episode_active[:] = True
        env._reset_level[:] = 0
        env._primary_reset[:] = True
        env.timed_out[:] = True
        env.terminated[:] = False
        env._best_hold[:] = 4.9
        env._curriculum_episodes = 62
        env._curriculum_successes = 0
        env._record_handstand_episodes(torch.ones(2, dtype=torch.bool))
        assert env.curriculum_level == 0
    finally:
        backend.close()


@torch.inference_mode()
def test_common_success_mask_agrees_with_native_support(bank):
    policy = Policy(BALANCE, "go2balance", 2, True)
    try:
        ids = bank.indices(0)[:2]
        samples = [sample_from_arrays(bank.data, int(i)) for i in ids]
        install_samples(policy, samples)
        valid, supported = valid_recovery(policy.env, np.zeros(2, dtype=bool))
        assert (valid <= supported).all()
        # Prior collision invalidates even otherwise valid front support.
        valid, _ = valid_recovery(policy.env, np.ones(2, dtype=bool))
        assert not valid.any()
    finally:
        policy.env._backend.close()


def test_frozen_labels_describe_snapshot_replays(bank):
    rows = json.loads((ROOT / "train_labels.json").read_text())
    assert len(rows) == len(bank.data["seed"])
    assert all(row["frozen"]["validity"] == "valid" for row in rows)
    assert all(
        row["frozen"]["hold_s"] >= 5 - 1e-6 for row in rows if row["frozen"]["success"]
    )
    assert any(row["kind"] == "standing_kick" for row in rows)
    assert any(row["source_seed"] == 6003 for row in rows)


@torch.no_grad()
def test_resumed_curriculum_supports_inference_mode_burn_in(bank):
    cfg, train = frontier_configuration(ROOT / "train.npz")
    cfg.env.num_envs = 2
    cfg.seed = 7
    train.runner.device = "cpu"
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    backend = MuJocoCPUBackend()
    env = Go2KickupRecovery(cfg, "cpu", True, backend)
    try:
        runner = task_registry.make_alg_runner(env, train)
        env.load_training_state_dict(env.training_state_dict())
        # This is the resume launcher's wrapper; plain no-grad burn-in fails
        # because reset/rollout may replace buffers with inference tensors.
        torch.inference_mode()(runner.burn_in_normalization)(n_iterations=2)
        assert torch.isfinite(env.root_states).all()
    finally:
        backend.close()
