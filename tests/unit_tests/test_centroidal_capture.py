import numpy as np
from experiments.centroidal_capture.core import capture_quality, correction, modify


def example():
    return dict(
        pitch=np.pi / 2,
        com=[0, 0, 0.5600034657639977],
        com_velocity=[0, 0, 0],
        omega=[0, 0, 0],
        H=[0, 0, 0],
        gaps=[0, 0, 0.5, 0.5],
        foot_velocity=np.zeros((4, 3)),
        com_foot_xy=[0, 0],
        front_loaded=True,
        both_front=True,
        rear_clearance=0.5,
    )


def test_capture_objective_prefers_support_and_regulation():
    x = example()
    q = capture_quality(x)
    assert q > capture_quality(dict(x, front_loaded=False, both_front=False))
    assert q > capture_quality(dict(x, com_velocity=[0, 0, -2], H=[0, 2, 0]))
    assert q > capture_quality(dict(x, pitch=np.deg2rad(35)))
    assert q > capture_quality(dict(x, rear_clearance=0))


def test_independent_front_corrections_and_bounds():
    names = [
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_thigh_joint",
        "RR_calf_joint",
    ]
    out = modify(np.zeros(6), names, np.array([-2, 0.2, 0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_allclose(out, [-1, 0.2, 0.3, 0.4, 0.5, 0.6])
    params = np.arange(30).reshape(5, 6)
    np.testing.assert_allclose(correction(params, 0), params[0])
    np.testing.assert_allclose(correction(params, 0.8), params[-1])
    np.testing.assert_allclose(correction(params, 0.08), (params[0] + params[1]) / 2)
