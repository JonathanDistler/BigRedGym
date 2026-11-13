"""Retained-skill distillation and source-guarded reuse of serial unified PPO."""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/retained_go2.py").resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
)
import torch
import scripts.unified_go2 as runner
from experiments.unified_go2.core import load, digest, save_json, source_contract
from experiments.retained_go2.actor import Student, FEATURES, FLOOR
from experiments.front_catch.io import atomic_checkpoint
import experiments.retained_go2.learning as learning


def contract():
    result = source_contract()
    for p in [
        *Path("experiments/retained_go2").glob("*.py"),
        Path("scripts/retained_go2.py"),
    ]:
        result[str(p)] = digest(p)
    for root in [
        Path("resources/robots/go2"),
        Path("gym/envs/go2"),
        Path("experiments/morphology"),
    ]:
        for p in root.glob("**/*"):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix in [".py", ".urdf", ".xml", ".stl", ".obj"]
            ):
                result[str(p)] = digest(p)
    return result


def distill(a, cfg, data, assets, bank):
    torch.manual_seed(a.seed)
    net = Student.from_adc(
        load(a.preparation / "adc.pt"), load(a.preparation / "balance.pt")
    )
    obs = torch.tensor(data["observations"])
    target = torch.tensor(data["actions"])
    roles = torch.tensor(data["roles"])
    groups = [torch.where(roles == i)[0] for i in [0, 1]]
    # The gate is fitted only to competent frames; no labels in the bridge gap.
    f = obs[:, FEATURES]
    bal = f[groups[1]]
    floor = torch.tensor(FLOOR, dtype=torch.float32)
    with torch.no_grad():
        net.actor.center.copy_(f.mean(0))
        net.actor.scale.copy_(f.std(0, unbiased=False).clamp(min=floor))
        width = (bal.amax(0) - bal.amin(0)) * 0.2 + floor
        net.actor.inner_low.copy_(bal.amin(0) - width)
        net.actor.inner_high.copy_(bal.amax(0) + width)
        net.actor.outer_width.copy_(width)
    from experiments.unified_go2.core import adc_network

    reference = adc_network(load(a.preparation / "adc.pt"))
    with torch.no_grad():
        initial_error = float(
            (net.actor(obs).tanh() - reference.actor(obs).tanh()).abs().max()
        )
    if initial_error > 2e-6:
        raise RuntimeError("ADC initialization changed actions")
    opt = torch.optim.Adam(net.actor.router.parameters(), lr=0.02)
    losses = []
    for step in range(a.steps):
        ids = torch.cat([g[torch.randint(len(g), (64,))] for g in groups])
        labels = roles[ids] * 2 - 1
        score = net.actor.router((f[ids] - net.actor.center) / net.actor.scale).squeeze(
            -1
        )
        loss = (2 - labels * score).clamp(
            min=0
        ).square().mean() + 1e-5 * net.actor.router.weight.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            losses.append(dict(step=step, competence_margin_loss=float(loss.detach())))
    net.actor.enabled.fill_(1)
    with torch.no_grad():
        predictions = net.actor(obs).tanh()
        gates = net.actor.gate(obs)
        by_role = {}
        for role, ids in enumerate(groups):
            by_role[str(role)] = dict(
                frames=len(ids),
                action_mse=float((predictions[ids] - target[ids]).square().mean()),
                per_joint_mse=(predictions[ids] - target[ids])
                .square()
                .mean(0)
                .tolist(),
                gate_min=float(gates[ids].min()),
                gate_max=float(gates[ids].max()),
            )
    payload = dict(
        network=net.state_dict(),
        distillation_optimizer=opt.state_dict(),
        steps=a.steps,
        seed=a.seed,
        preparation_sha=digest(a.preparation / "manifest.json"),
        source_contract=contract(),
        architecture="ADC-preserving internal competence-gated balance specialization + zero shared correction",
    )
    atomic_checkpoint(a.output / "student.pt", payload)
    save_json(
        a.output / "distillation.json",
        dict(
            initial_ADC_max_action_error=initial_error,
            normalization="Expert inputs unchanged; only router has fixed saved center/scale buffers",
            by_role=by_role,
            losses=losses,
            competence="Demonstrated skill frames only; compact support estimated from successful balance data, no fixed pitch/time threshold",
        ),
    )
    print("distillation", initial_error, by_role, flush=True)


original_train = runner.train


def guarded_train(a, *args, **kwargs):
    import json

    certificate = json.loads(CERTIFICATE.read_text())
    if not certificate["passed"]:
        raise RuntimeError("Endpoint retention has not passed; PPO is blocked")
    provenance = a.preparation / "bridge_provenance.json"
    if digest(a.preparation / "manifest.json") == certificate["preparation_sha"]:
        pass  # A fresh host may validate directly on the final bridge preparation.
    elif provenance.exists():
        proof = json.loads(provenance.read_text())
        if not (
            proof["endpoint_configs_identical"]
            and proof["competent_demonstrations_identical"]
            and proof["endpoint_balance_states_unchanged"]
            and proof["source_manifest_sha"] == certificate["preparation_sha"]
        ):
            raise ValueError("Bridge preparation changed endpoint conventions")
    else:
        raise ValueError("Unvalidated preparation")
    import numpy as np

    cfg = args[0]
    if (
        cfg.kp != 60.0
        or not np.allclose(cfg.kd, 3.0)
        or cfg.dt != 0.02
        or cfg.action_scale != 2.0
    ):
        raise ValueError(
            "This experiment requires the verified nominal Go2 control convention"
        )
    learning.configure(a, certificate)
    return original_train(a, *args, **kwargs)


CERTIFICATE = Path("output/retained_go2_validation_v3/retention.json")
runner.Student = Student
runner.distill = distill
runner.source_contract = contract
runner.Vector = learning.Vector
runner.evaluate = learning.evaluate
runner.promote = learning.promote
runner.train = guarded_train
if __name__ == "__main__":
    if "--certificate" in sys.argv:
        index = sys.argv.index("--certificate")
        CERTIFICATE = Path(sys.argv[index + 1])
        del sys.argv[index : index + 2]
    runner.main()
