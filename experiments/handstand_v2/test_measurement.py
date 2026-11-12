import math
import numpy as np
from experiments.handstand_v2.measurement import Measurement, shaping


def row(pitch=36, loaded=True, illegal=""):
    return dict(
        pitch=math.radians(pitch),
        pitch_velocity=0.2,
        loads=np.array([10, 10, 0, 0] if loaded else [0, 0, 0, 0]),
        weight=100,
        feet=np.zeros((4, 3)),
        rear_clearance=0.04,
        front_drift=0.0,
        illegal_bodies=illegal,
        self_contacts=0,
        com_velocity=np.zeros(3),
        front_support=loaded,
    )


def test_crossing_survives_later_fall():
    m = Measurement()
    m.observe(row(), 0.1, 0.002, True, True, 35)
    m.observe(row(illegal="FL_calf"), 0.2, 0.002, False, True, 35)
    assert 35 in m.crossings and m.first_invalid["time"] == 0.2
    assert m.first_invalid["bodies"] == "FL_calf"


def test_recoverable_unloading_gap_persistence_reset():
    m = Measurement()
    for i in range(30):
        m.observe(row(loaded=False), i * 0.002, 0.002, True, True, 35)
    for i in range(12):
        m.observe(row(), 0.1 + i * 0.002, 0.002, True, True, 35)
    assert m.gap == 0 and m.max_gap > 0.02 and m.recovered and 35 in m.crossings
    assert Measurement().max_gap == 0


def test_loaded_slip_and_flight_displacement():
    for loaded in (False, True):
        m = Measurement()
        r = row(loaded=loaded)
        m.observe(r, 0, 0.002, True, True, 35)
        r["feet"][:2, 0] = 0.03
        r["ground_slip_speed"] = 15.0 if loaded else 0.0
        m.observe(r, 0.002, 0.002, True, True, 35)
        assert (m.slip > 0.02) == loaded
        if loaded:
            assert m.failed is False and m.certified_peak == math.radians(36)


def test_calf_disqualifies_and_onset_is_exact():
    m = Measurement()
    m.observe(row(25), 0.1, 0.002, True, True, 35)
    m.observe(row(illegal="RR_calf"), 0.12, 0.002, False, True, 35)
    m.observe(row(45), 0.14, 0.002, True, True, 35)
    assert m.first_invalid["time"] == 0.12 and 35 not in m.crossings


def test_dense_progress_no_flight_or_unsafe_credit():
    for pitch in (26, 30, 35, 40, 45):
        r = row(pitch)
        assert shaping(r, True, True, 45, 0.002)["progress"] > 0
        assert sum(shaping(r, False, True, 45, 0.002).values()) == 0
        r["front_support"] = False
        assert sum(shaping(r, True, True, 45, 0.002).values()) == 0


def test_rolling_center_is_not_material_slip():
    m = Measurement()
    r = row()
    r["ground_slip_speed"] = 0.0
    m.observe(r, 0, 0.002, True, True, 35)
    r["feet"][:2, 0] = 0.03
    m.observe(r, 0.002, 0.002, True, True, 35)
    assert m.slip == 0.0


def test_motor_failure_cannot_receive_progress():
    r = row(40)
    assert shaping(r, False, True, 45, 0.002)["progress"] == 0
    m = Measurement()
    m.observe(r, 0.1, 0.002, False, True, 35)
    assert not m.crossings


def test_native_interface_contact_forces_and_reward(tmp_path):
    import torch
    import curriculum
    from experiments.morphology.learning.environment import Assets
    from experiments.handstand_v2.environment import V2Environment

    torch.set_num_threads(1)
    checkpoint = (
        "output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt"
    )
    net, hp, cfg, _ = curriculum.load_source(checkpoint)
    env = V2Environment(Assets(tmp_path), cfg)
    a, c = env.reset(1.0, 70400)
    assert a.shape == (44,) and c.shape == (74,)
    useful = 0
    while True:
        with torch.no_grad():
            action = net.actor(torch.tensor(a)).tanh().numpy()
        (a, c), reward, done, info = env.step(action)
        assert np.isfinite(reward) and np.isfinite(a).all() and np.isfinite(c).all()
        useful += info.get("reward_v2_progress", 0)
        if done:
            break
    m = env.measurement
    assert m.first_invalid["time"] > m.peak_time
    assert m.first_invalid["bodies"] == "FL_calf"
    assert m.certified_peak > math.radians(25)
    assert m.certified_peak <= m.loaded_peak
    assert m.slip >= 0 and useful > 0
    assert info["takeoff"]
    assert all(r["v2_progress"] == 0 for r in env.trace if not r["support_valid"])
    env.reset(1.0, 70400)
    assert env.measurement.first_invalid is None and env.measurement.max_gap == 0


def test_dynamic_flight_crossing_is_independent_of_certification():
    m = Measurement()
    r = row(45, loaded=False)
    m.observe(r, 0.1, 0.002, True, True, 45)
    assert 45 in m.dynamic_crossings and not m.dynamic_crossings[45]["loaded"]
    assert 45 not in m.crossings and m.certified_peak == 0


def test_catch_starts_after_crossing_and_survives_later_fall():
    m = Measurement()
    for i in range(20):
        m.observe(row(34), i * 0.002, 0.002, True, True, 35)
    assert m.best_catch == 0
    for i in range(100):
        m.observe(row(36), 0.1 + i * 0.002, 0.002, True, True, 35)
    assert m.best_catch >= 0.15
    m.observe(row(illegal="FL_calf"), 0.4, 0.002, False, True, 35)
    assert m.best_catch >= 0.15 and 35 in m.crossings


def test_front_recovery_independent_of_target_and_handstand():
    m = Measurement()
    m.observe(row(loaded=False), 0, 0.002, True, True, 90)
    for i in range(15):
        m.observe(row(30), 0.1 + i * 0.002, 0.002, True, True, 90)
    assert m.recovered and m.best_front_recovery >= 0.02
    assert m.best_catch == 0 and m.best_handstand == 0
    for i in range(20):
        m.observe(row(90), 0.2 + i * 0.002, 0.002, True, True, 90)
    assert m.best_handstand >= 0.02


def test_evaluation_preserves_training_rng(tmp_path):
    import torch
    import curriculum
    from experiments.handstand_v2.run import evaluate, ORIGINAL
    from experiments.morphology.learning.environment import Assets

    torch.set_num_threads(1)
    net, _, cfg, _ = curriculum.load_source(ORIGINAL)
    assets = Assets(tmp_path / "assets")
    state = torch.get_rng_state().clone()
    evaluate(net, assets, cfg, "rng_probe", tmp_path, [70400], plot=False)
    assert torch.equal(torch.get_rng_state(), state)


def test_fast_dynamic_passage_is_not_controlled_crossing():
    m = Measurement()
    r = row(45)
    r["pitch_velocity"] = 4.0
    m.observe(r, 0.1, 0.002, True, True, 45)
    assert 45 in m.dynamic_crossings and m.certified_peak > 0
    assert 45 not in m.crossings and m.best_catch == 0
