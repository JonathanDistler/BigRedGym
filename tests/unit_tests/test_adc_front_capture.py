import copy
import numpy as np
import pytest
import torch
from experiments.adc_front_capture.curriculum import (
    ActuatorCurriculum,
    damping,
    effective_inertia,
)
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.adc_front_capture.mechanics import region_distance, measurements, reach
from experiments.front_catch.core import CatchConfig
from experiments.front_catch.vector import Vector
from experiments.morphology.learning.environment import Assets


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("adc_assets"))


def test_schedule_and_nominal_endpoint():
    c = ActuatorCurriculum(initial_k=75, alpha=1, max_seconds=8)
    c.update([dict(episode_seconds=4)], 1)
    assert c.stiffness == 67.5
    c.update([dict(episode_seconds=8)], 2)
    assert c.stiffness == 60
    c.update([dict(episode_seconds=1)], 3)
    assert c.stiffness == 60
    with pytest.raises(ValueError):
        ActuatorCurriculum(initial_k=100)


def test_recovery_cannot_complete_from_pitch_alone():
    c = ActuatorCurriculum(initial_k=75, alpha=1, rule="recovery")
    c.update([dict(peak_collision_free_pitch=np.pi / 2, handstand_seconds=0)], 1)
    assert c.progress == pytest.approx(0.35)
    c.update(
        [
            dict(
                peak_collision_free_pitch=np.pi / 2,
                handstand_seconds=5,
                valid_support_transition=True,
            )
        ],
        2,
    )
    assert c.progress == 1


def test_loaded_support_region():
    feet = np.array([[0, 0.1, 0.022], [0, -0.1, 0.022]])
    assert region_distance(np.zeros(3), feet, [1, 1]) == 0
    assert region_distance(np.zeros(3), feet, [1, 0]) > 0
    assert np.isnan(region_distance(np.zeros(3), feet, [0, 0]))


def test_gains_are_lane_local_and_nominal_exact(assets):
    cfg = CatchConfig(
        adc_enabled=True,
        capture_enabled=True,
        comparison_suite="anticipatory_v1",
        catch_mode="B",
    )
    with Vector(assets, {"D": cfg}, {}, 2, workers=0) as v:
        a, c = v.reset("D", [1, 0, 0, 0, 0, 0])
        v.dynamics("D", 75, 60, 3)
        assert v.lanes["D"][0].env.cfg.kp == 75
        state = copy.deepcopy(v.state("D"))
        actions = np.full((2, 12), 0.01)
        first = v.step("D", actions, [1, 0, 0, 0, 0, 0])
        v.restore("D", state)
        second = v.step("D", actions, [1, 0, 0, 0, 0, 0])
        for left, right in zip(first, second):
            assert np.array_equal(left[0], right[0]) and left[2] == right[2]
        v.dynamics("D", 60, 60, 3)
        assert np.all(v.lanes["D"][0].env.cfg.kd == 3)
        assert cfg.kp == 60


def test_kinematics_and_ik_do_not_mutate_state(assets):
    e = CaptureEnvironment(assets, CatchConfig(adc_enabled=True, capture_enabled=True))
    q = e.d.qpos.copy()
    v = e.d.qvel.copy()
    m = measurements(e)
    assert np.array_equal(q, e.d.qpos) and np.array_equal(v, e.d.qvel)
    assert all(np.isfinite(x) for x in effective_inertia(e))
    assert np.all(damping(e, 75) >= 0)
    assert "FL_knee_z" in m and "FR_body_x" in m
    ok, error, ratio = reach(e, "FL", e.d.geom_xpos[e.foot_geoms[0]].copy())
    assert error < 0.001
    ok, error, ratio = reach(e, "FL", np.array([4, 4, 0]))
    assert not ok


def test_no_standing_success_or_displacement_reward(assets):
    e = CaptureEnvironment(assets, CatchConfig(adc_enabled=True, capture_enabled=True))
    (a, c), reward, done, info = e.step(np.zeros(12))
    assert a.shape == (73,) and c.shape == (103,)
    assert not info["success"] and info["reward"] == reward
    assert "reward_useful_placement" not in e.trace[-1]
    assert e.trace[-1]["reward_pose"] == 0


def test_update_boundary_restores_optimizer_rng_and_curriculum(assets):
    from experiments.morphology.learning.ppo import ActorCritic, update
    from experiments.front_catch.learning import collect, hp_defaults

    torch.set_num_threads(1)
    cfg = CatchConfig(adc_enabled=True, capture_enabled=True)
    hp = hp_defaults()
    hp.update(bc_weight=0.0, epochs=1)
    mix = [1, 0, 0, 0, 0, 0]
    torch.manual_seed(123)
    net = ActorCritic(73, 103)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
    with Vector(assets, {"D": cfg}, {}, 2, workers=0) as vector:
        a, c = vector.reset("D", mix)
        vector.dynamics("D", 67.5, 60, 3)
        batch, a, c, _, _ = collect(net, vector, "D", a, c, 4, mix, hp)
        update(net, optimizer, batch, hp)
        boundary = copy.deepcopy(
            dict(
                network=net.state_dict(),
                optimizer=optimizer.state_dict(),
                world=vector.state("D"),
                torch_rng=torch.get_rng_state(),
                a=a,
                c=c,
            )
        )
        batch, next_a, _, _, _ = collect(net, vector, "D", a, c, 4, mix, hp)
        update(net, optimizer, batch, hp)
        expected = copy.deepcopy(net.state_dict())
        net.load_state_dict(boundary["network"])
        optimizer.load_state_dict(boundary["optimizer"])
        restored_a, restored_c = vector.restore("D", boundary["world"])
        assert np.array_equal(restored_a, boundary["a"])
        assert np.array_equal(restored_c, boundary["c"])
        torch.set_rng_state(boundary["torch_rng"])
        batch, actual_a, _, _, _ = collect(
            net, vector, "D", restored_a, restored_c, 4, mix, hp
        )
        update(net, optimizer, batch, hp)
        assert np.array_equal(next_a, actual_a)
        assert all(torch.equal(expected[k], net.state_dict()[k]) for k in expected)


def test_contact_curriculum_requires_recovery_not_initial_pitch():
    from experiments.adc_front_capture.curriculum import ContactCurriculum

    curriculum = ContactCurriculum()
    bank = {
        "train": {
            kind: []
            for kind in (
                "loaded",
                "pre_liftoff",
                "post_liftoff",
                "partial",
                "near_vertical",
            )
        }
    }
    bank["train"]["post_liftoff"] = [object()]
    assert curriculum.mixture(True, bank) == [0.35, 0.0, 0.0, 0.65, 0.0, 0.0]
    rows = [
        dict(evaluation_type="standing", certified_standing_takeoff=True),
        dict(
            evaluation_type="intermediate",
            peak_collision_free_pitch=np.pi / 2,
            handstand_seconds=0.0,
        ),
    ]
    curriculum.update(rows, 1)
    assert curriculum.stage == 0
    rows[1]["handstand_seconds"] = 0.2
    curriculum.update(rows, 2)
    assert curriculum.stage == 1
    assert curriculum.mixture(True, bank)[0] == 0.5
    rows[0]["certified_standing_takeoff"] = False
    curriculum.update(rows, 3)
    assert curriculum.stage == 1
