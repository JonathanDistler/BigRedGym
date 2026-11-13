"""Real imports, checkpoint loads and native model checks inside the new runtime."""

import importlib
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.molab.common import (
    ROOT,
    PYTHON,
    atomic_json,
    digest,
    load,
    stamp,
    verify_manifest,
)

MODULES = [
    "torch",
    "mujoco",
    "numpy",
    "pandas",
    "matplotlib",
    "tensordict",
    "wandb",
    "scripts.run_front_catch",
    "scripts.benchmark_front_catch",
    "scripts.report_front_catch",
    "scripts.report_front_catch_timing",
    "scripts.play_front_catch",
    "scripts.analyze_front_catch",
    "scripts.backup_front_catch",
    "experiments.front_catch.anticipatory",
    "experiments.front_catch.learning",
    "experiments.front_catch.vector",
    "experiments.morphology.learning.ppo",
    "gym.envs.go2.leg_morphology",
]


def main():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "Validation must use " + str(PYTHON) + "; got " + sys.executable
        )
    manifest = verify_manifest()
    for module in MODULES:
        importlib.import_module(module)
    import torch
    import numpy as np
    from experiments.front_catch.data import network, config, validate_state
    from experiments.front_catch.anticipatory import environment_class
    from experiments.morphology.learning.environment import Assets

    torch.set_num_threads(1)
    prep = ROOT / "output/front_catch_preparation_20261112_v3"
    prepared = load(prep / "manifest.json")
    initial = ROOT / prepared["initial"]
    if digest(initial) != prepared["initial_sha256"]:
        raise ValueError("Preserved initialization checkpoint changed")
    net, parent = network(initial)
    for key, filename in [
        ("bank_sha256", "reset_bank.pt"),
        ("dataset_sha256", "bc_demonstrations.npz"),
    ]:
        if digest(prep / filename) != prepared[key]:
            raise ValueError("Preparation checksum mismatch: " + filename)
    bank = torch.load(prep / "reset_bank.pt", weights_only=False, map_location="cpu")
    demos = np.load(prep / "bc_demonstrations.npz")
    assert demos["observations"].shape[1] == 73 and demos["actions"].shape[1] == 12
    with tempfile.TemporaryDirectory(dir=ROOT / ".molab") as assets_dir:
        cfg = config(parent, "C")
        cfg.comparison_suite = "anticipatory_v1"
        env = environment_class(cfg)(Assets(assets_dir), cfg)
        assert abs(env.mass - prepared["model_mass"]) < 1e-9
        assert validate_state(env)[0]
        obs, critic = env.observations()
        with torch.no_grad():
            action = net.actor(torch.tensor(obs)).tanh().numpy()
        assert action.shape == (12,) and np.isfinite(action).all()
        _, reward, _, _ = env.step(action)
        assert np.isfinite(reward)
        for split in ("train", "evaluation"):
            for kind, states in bank[split].items():
                assert states, (split, kind)
                env.restore_start(states[0]["state"], kind)
                assert validate_state(env)[0], (split, kind)
        mass = env.mass
    packages = {}
    for line in (ROOT / "deployment/molab/requirements.lock").read_text().splitlines():
        if "==" not in line or line.startswith(("#", " ", "--")):
            continue
        name, pinned = line.split("==", 1)
        actual = version(name)
        if actual != pinned:
            raise RuntimeError(
                f"Dependency version mismatch: {name}: {actual} != {pinned}"
            )
        packages[name] = actual
    local_imports = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).is_file() and Path(file).resolve().is_relative_to(ROOT):
            relative = str(Path(file).resolve().relative_to(ROOT))
            if (
                relative.endswith(".py")
                and not relative.startswith(".venv311/")
                and not relative.startswith(".python/")
            ):
                if relative not in manifest["files"]:
                    raise RuntimeError("Unpackaged internal import: " + relative)
                local_imports[name] = relative
    report = dict(
        timestamp=stamp(),
        python=sys.version,
        executable=sys.executable,
        real_interpreter=str(Path(sys.executable).resolve()),
        packages=packages,
        modules=MODULES,
        internal_imports=local_imports,
        initialized_actor_actions=12,
        initial_sha256=digest(initial),
        mass=mass,
        manifest_files=len(manifest["files"]),
        source_revision=manifest["source_revision"],
        scheduler=(
            "No PPO scheduler exists in the unchanged trainer; "
            "optimizer state is checkpointed."
        ),
    )
    atomic_json(ROOT / ".molab/environment.json", report)
    print(
        "VALIDATION SUCCESS:",
        len(packages),
        "pinned packages;",
        len(local_imports),
        "internal imports; native model, BC checkpoint and reset bank validated.",
    )


if __name__ == "__main__":
    main()
