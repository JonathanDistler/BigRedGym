"""Bridge sampling is reproducible and leaves the deployed actor untouched."""

import copy
import numpy as np
import torch
from scripts.unified_go2 import setup
from experiments.dynamic_bridge.sampling import Vector
from experiments.retained_go2.actor import Student
from experiments.unified_go2.core import load
from experiments.front_catch.data import validate_state
from pathlib import Path

PREP = Path("output/retained_go2_preparation_v3")


def test_bridge_sampling_and_native_restore(tmp_path):
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(PREP, tmp_path)
    v = Vector(assets, cfg, bank, 1, data["centers"], seed=84)
    p = v.bridge_probabilities()
    assert np.isclose(p.sum(), 1) and np.all(p >= 0.2 / len(p))
    v.start(0, "bridge")
    assert validate_state(v.envs[0])[0]
    state = copy.deepcopy(v.state())
    first = [v.start(0, "bridge")[0] for _ in range(3)]
    v.restore(state)
    second = [v.start(0, "bridge")[0] for _ in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(v.bridge_failure, state["dynamic_bridge"]["failure"])


def test_no_actor_or_teacher_convention_change():
    from experiments.unified_go2.core import digest

    model = Path("output/retained_go2_distillation_v3/student.pt")
    assert (
        digest(model)
        == "0d08913c5f306f4090f62f74001a62a795feee533958e3561adcfee816c984ab"
    )
    net = Student()
    net.load_state_dict(load(model)["network"])
    data = np.load(PREP / "demonstrations.npz")
    with torch.no_grad():
        out = net.actor(torch.tensor(data["observations"][:8])).tanh().numpy()
    assert np.max(abs(out - data["actions"][:8])) < 2e-6


def test_continuation_rejects_missing_evidence_and_stop(tmp_path):
    import pytest
    from scripts.continue_dynamic_bridge import permission

    with pytest.raises(ValueError, match="completed"):
        permission(tmp_path, 25)
    (tmp_path / "STOP").write_text("Endpoint regression")
    with pytest.raises(ValueError, match="STOP"):
        permission(tmp_path, 100)
