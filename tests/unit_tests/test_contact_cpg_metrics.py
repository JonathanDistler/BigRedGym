"""Exercise optional CPG contacts and saved diagnostics with real CPU physics."""

import csv
import json

import pytest
import torch

from gym.envs.go2.go2handstand import Go2Handstand
from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg
from gym.utils.robot_metrics import RobotMetrics, configure_contact_cpg
from types import SimpleNamespace


def test_cpg_modes_forces_resets_and_saved_metrics(tmp_path):
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.task_registry import task_registry

    cfg = Go2HandstandCfg()
    runner = Go2HandstandRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    configure_contact_cpg(
        SimpleNamespace(
            cpg=True, contact_modes=["stance", "stance", "swing", "ignore"]
        ),
        cfg,
        runner,
    )
    task_registry.convert_frequencies_to_params(cfg, runner)
    backend = MuJocoCPUBackend()
    try:
        env = Go2Handstand(cfg, "cpu", True, backend)
        assert env.cpg_contact_obs[0].tolist() == [1, 1, 0, -1]
        assert torch.isfinite(env.get_states(runner.actor.obs)).all()
        env.contact_forces.zero_()
        weight = 9.81 * env._backend.link_mass.sum(dim=1)
        env.contact_forces[:, env.feet_indices[:2], 2] = weight[:, None] / 2
        assert env._reward_cpg_contact().tolist() == pytest.approx([1, 1])
        # Ignored rear-right contact cannot change reward; swing contact must.
        env.contact_forces[:, env.feet_indices[3], 2] = weight
        assert env._reward_cpg_contact().tolist() == pytest.approx([1, 1])
        env.contact_forces[:, env.feet_indices[2], 2] = weight
        assert (env._reward_cpg_contact() < 1).all()
        # Force feedback makes a loaded clock differ from a freely running clock.
        env.leg_phase.zero_()
        env._cpg_load.zero_()
        env.contact_forces.zero_()
        env.contact_forces[0, env.feet_indices[0], 2] = weight[0]
        env._post_physics_step()
        assert env.leg_phase[0, 0] < env.leg_phase[1, 0]
        saved = env.leg_phase[1].clone()
        env._reset_idx(torch.tensor([True, False]))
        assert torch.equal(env.leg_phase[1], saved)
        assert torch.allclose(env.leg_phase[0], env._cpg_offsets)
        metrics = RobotMetrics(env, tmp_path, num_envs=2)
        env.contact_forces.zero_()
        env.contact_forces[:, env.feet_indices[:2], 2] = 50
        metrics.record()
        env.step()
        metrics.record()
        metrics.close()
        rows = list(csv.DictReader((tmp_path / "robot_metrics.csv").open()))
        assert len(rows) == 4
        assert float(rows[0]["front_force_symmetry"]) == 1
        assert float(rows[0]["FL_foot_force_z_N"]) == 50
        assert rows[0]["rear_force_symmetry"] == ""
        assert rows[0]["RL_foot_desired_contact"] == "0.0"
        assert "FL_hip_joint_angle_rad" in rows[0]
        assert json.loads((tmp_path / "summary.json").read_text())["rows"] == 4
    finally:
        backend.close()
