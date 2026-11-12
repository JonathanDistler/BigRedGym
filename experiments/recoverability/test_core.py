import numpy as np
import pytest
import torch
from experiments.recoverability.physics import World, Metrics, feasible_state
from experiments.recoverability.controllers import (
    SavedController,
    BalanceAdapter,
    CANDIDATES,
)
from experiments.morphology.learning.environment import Assets
from scripts.evaluate_kickup_handoff import sync
import curriculum


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    torch.set_num_threads(1)
    assets = Assets(tmp_path_factory.mktemp("recovery_assets"))
    _, _, cfg, _ = curriculum.load_source(
        "output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt"
    )
    c = SavedController(CANDIDATES[1][1], "go2balance")
    yield assets, cfg, c
    c.close()


def test_adapter_observation_and_pd_parity(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    e = c.env
    b = e._backend
    d = b._datas[0]
    w.d.qpos[:] = d.qpos
    w.d.qvel[:] = d.qvel
    import mujoco

    mujoco.mj_forward(w.m, w.d)
    w.target = (e._standing_posture() + e.dof_pos_target)[0].numpy().astype(float)
    a = BalanceAdapter(c, w)
    a.phase = e.leg_phase[0].clone()
    a.load = e._cpg_load[0].clone()
    a.previous = e.dof_pos_target[0].clone()
    e.handstand_stage[:] = 2
    sync(e)
    native = c.runner.get_obs(c.runner.actor_cfg["obs"])
    assert a.observation().shape == (1, 59)
    torch.testing.assert_close(a.observation(), native, atol=3e-5, rtol=2e-5)
    with torch.no_grad():
        expected = c.actor.act_inference(native)
    target = a.target()
    torch.testing.assert_close(
        torch.tensor(target, dtype=torch.float32),
        e._standing_posture()[0] + 0.25 * expected[0],
        atol=1e-5,
        rtol=1e-5,
    )
    c.runner.set_actions(c.runner.actor_cfg["actions"], expected, False)
    torque = e._compute_torques()[0].numpy()
    physical = np.clip(
        cfg.kp * (target - w.d.qpos[w.qids]) - cfg.kd * w.d.qvel[w.vids],
        -w.effort,
        w.effort,
    )
    np.testing.assert_allclose(physical, torque, atol=3e-4, rtol=2e-5)


def test_feasible_and_invalid_proposals(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    ok, details = feasible_state(w, c.env._standing_posture()[0].numpy(), 90001)
    assert ok, details
    w.d.qpos[w.qids[0]] = w.bounds[0, 1] + 0.1
    import mujoco

    mujoco.mj_forward(w.m, w.d)
    assert not w.screen()[0]


def test_switch_memory_does_not_change_physical_state(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    feasible_state(w, c.env._standing_posture()[0].numpy(), 90002)
    q, v, t = w.d.qpos.copy(), w.d.qvel.copy(), w.d.time
    adapter = BalanceAdapter(c, w)
    adapter.target()
    np.testing.assert_array_equal(w.d.qpos, q)
    np.testing.assert_array_equal(w.d.qvel, v)
    assert w.d.time == t


def test_gap_reset_and_failure_persistence(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    feasible_state(w, c.env._standing_posture()[0].numpy(), 90003)
    m = Metrics(w)
    row = w.physical_row()
    row["front_support"] = True
    m.observe(row, 0.002)
    row["front_support"] = False
    m.observe(row, 0.002)
    assert m.gap > 0
    row["front_support"] = True
    m.observe(row, 0.002)
    assert m.gap == 0 and m.max_gap > 0


def test_cpg_memory_matches_native_physics_update(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    e = c.env
    a = BalanceAdapter(c, w)
    a.phase = e.leg_phase[0].clone()
    a.load = e._cpg_load[0].clone()
    loads = e.contact_forces[0, e.feet_indices, 2].clamp(min=0).numpy()
    a.advance_memory(loads, w.m.opt.timestep)
    e._post_physics_step()
    torch.testing.assert_close(a.phase, e.leg_phase[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a.load, e._cpg_load[0], atol=2e-6, rtol=2e-6)


def test_kick_metrics_are_observed_at_every_physics_step(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    m = Metrics(w)
    w.monitor = m
    calls = []
    original = m.observe

    def observe(row, dt):
        calls.append(w.d.time)
        return original(row, dt)

    m.observe = observe
    w.step(np.zeros(12))
    assert len(calls) == round(cfg.dt / w.m.opt.timestep)
    assert np.all(np.diff(calls) > 0)


@pytest.mark.parametrize("roll", [-2.0, 2.0])
def test_roll_proposal_has_feasible_bilateral_ground_height(fixture, roll):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    ok, details = feasible_state(
        w, c.env._standing_posture()[0].numpy(), 90004, roll=roll
    )
    assert ok, details
    np.testing.assert_allclose(w.d.geom_xpos[w.foot_geoms[:2], 2], 0.0222, atol=1e-5)


def test_counterfactual_does_not_mutate_live_state_or_clock(fixture):
    from experiments.recoverability.run import counterfactual_recovery

    assets, cfg, c = fixture
    w = World(assets, cfg)
    feasible_state(w, c.env._standing_posture()[0].numpy(), 90005)
    q, v, t = w.d.qpos.copy(), w.d.qvel.copy(), w.d.time
    result = counterfactual_recovery(c, w, assets, cfg, seconds=0.02)
    assert not result["success"]
    np.testing.assert_array_equal(q, w.d.qpos)
    np.testing.assert_array_equal(v, w.d.qvel)
    assert t == w.d.time


def test_initial_contact_acquisition_is_not_support_recovery(fixture):
    assets, cfg, c = fixture
    w = World(assets, cfg)
    feasible_state(w, c.env._standing_posture()[0].numpy(), 90006)
    m = Metrics(w)
    row = w.physical_row()
    row["front_support"] = False
    m.observe(row, 0.01)
    row["front_support"] = True
    for _ in range(10):
        m.observe(row, 0.002)
    assert not m.had_gap and not m.recovered and m.max_gap == 0


def test_47_dimensional_saved_source_observation_adapter(fixture):
    assets, cfg, _ = fixture
    c = SavedController(CANDIDATES[3][1], "go2reachability")
    try:
        w = World(assets, cfg)
        e = c.env
        d = e._backend._datas[0]
        w.d.qpos[:] = d.qpos
        w.d.qvel[:] = d.qvel
        import mujoco

        mujoco.mj_forward(w.m, w.d)
        w.target = (e._standing_posture() + e.dof_pos_target)[0].numpy().astype(float)
        a = BalanceAdapter(c, w)
        a.previous = e.dof_pos_target[0].clone()
        e.handstand_stage[:] = 2
        sync(e)
        assert a.observation().shape == (1, 47)
        torch.testing.assert_close(
            a.observation(),
            c.runner.get_obs(c.runner.actor_cfg["obs"]),
            atol=3e-5,
            rtol=2e-5,
        )
    finally:
        c.close()


def test_nominal_counterfactual_really_balances_five_seconds(fixture):
    from experiments.recoverability.run import counterfactual_recovery

    assets, cfg, c = fixture
    w = World(assets, cfg)
    accepted, details = feasible_state(w, c.env._standing_posture()[0].numpy(), 90007)
    assert accepted, details
    result = counterfactual_recovery(c, w, assets, cfg)
    assert result["success"] and result["hold"] >= 5.0
    assert result["max_torque"] <= 1.00001
