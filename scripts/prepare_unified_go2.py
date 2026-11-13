"""Discover and protect experts; finite native capture probes; competent labels only."""

import argparse, os, sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled")
import numpy as np, torch
from experiments.unified_go2.core import (
    load,
    digest,
    save_json,
    Environment,
    adc_network,
    descriptor,
)
from experiments.multi_impulse.core import MultiConfig
from experiments.front_catch.data import validate_state
from experiments.balance_feedback.actor import BalanceActor
from experiments.morphology.learning.environment import Assets
from experiments.front_catch.data import raw_state
from experiments.recoverability.physics import World, feasible_state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    # Discover by structure and validate contents, not a guessed unchecked filename.
    adc_candidates = list(Path("output").glob("**/B_best.pt"))
    balance_candidates = [
        x
        for x in Path("output").glob("balance_feedback*/balance_actor.pt")
        if "metadata" in load(x)
    ]
    if len(adc_candidates) != 1 or len(balance_candidates) != 1:
        raise ValueError("Ambiguous experts; select explicitly before proceeding")
    adc_path, balance_path = adc_candidates[0], balance_candidates[0]
    parent = load(adc_path)["runs"]["B_seed42"]
    balance = load(balance_path)
    cfg = MultiConfig(**balance["config"])
    cfg.kp = 60.0
    cfg.kd = 3.0
    cfg.seconds = 6.2
    cfg.corrective = True
    cfg.balance_training = False
    cfg.full_trace = True
    for k in ["dt", "action_scale", "kp"]:
        assert np.allclose(parent["config"][k], getattr(cfg, k))
    assert np.allclose(parent["config"]["kd"], cfg.kd)
    torch.save(parent, a.output / "adc.pt")
    shutil.copy2(balance_path, a.output / "balance.pt")
    assets = Assets(a.output / "assets")
    e = Environment(assets, cfg)
    e.reset(1, 1)
    net = adc_network(parent)
    expert = BalanceActor()
    expert.load_state_dict(balance["network"])
    refs = {
        str(p): digest(p)
        for p in [
            adc_path,
            balance_path,
            *Path("output/multi_impulse_study_v2").glob("checkpoint_00005*.pt"),
        ]
    }
    meta = dict(
        originals=refs,
        adc_architecture="73→256→256→128→12 ELU, tanh actions; critic103",
        student_architecture="Affine observable feedback plus quadratic-gravity residual MLP93→256→256→128→12, bounded position actions; critic123",
        balance_architecture="34 selected existing channels→linear12, clamped actions",
        normalization="No running normalization; deterministic physical scales in observation source; no statistics mixed",
        prefix_mapping="ADC residual/critic weights copied with20 extra input columns zero; balance affine coefficients embedded in93-channel actor",
        observation_order="gravity0:3, body_omega3:6, q-nominal6:18, qvel/speed18:30, target30:42, clock42(/8s), scale43, old phase44:47, momentum47:61, contacts61:73, state phase73:93",
        joint_names=e.names,
        nominal=e.nominal,
        joint_limits=e.bounds,
        torque_limits=e.effort,
        speed_limits=e.speed,
        dt=cfg.dt,
        kp=cfg.kp,
        kd=cfg.kd,
        action_scale=cfg.action_scale,
        config=vars(cfg),
    )
    save_json(a.output / "compatibility.json", meta)
    bank = {
        "train": {"bridge": [], "balance": []},
        "evaluation": {"bridge": [], "balance": []},
    }
    records = []
    obs_data = []
    action_data = []
    roles = []
    centers = []

    def probe(label, split, state, details):
        obs, _ = e.restore_start(state, "near_vertical")
        initial = obs.copy()
        valid, why = validate_state(e)
        if not valid:
            raise ValueError(why)
        frames = []
        while True:
            with torch.no_grad():
                action = expert(torch.tensor(obs)).numpy()
            frames.append((obs.copy(), action.copy()))
            (obs, _), _, done, info = e.step(action)
            if done:
                break
        success = bool(e.ever_hold_success)
        records.append(
            dict(
                label=label,
                split=split,
                initial_valid=True,
                success=success,
                hold=e.multi_best_hold,
                termination=info["termination"],
                **details,
            )
        )
        if success:
            bank[split]["balance"].append(state)
            if split == "train":
                centers.append(descriptor(initial))
                for x, y in frames:
                    obs_data.append(x)
                    action_data.append(y)
                    roles.append(1)
        elif e.corrections.peak > np.deg2rad(45):
            # Valid initial state only; failure is recorded and receives NO teacher labels.
            bank[split]["bridge"].append(state)
        print(records[-1], flush=True)

    oldbank = load("output/multi_impulse_preparation_v1/balance_bank.pt")
    for i, item in enumerate(oldbank["train"]["near_vertical"]):
        probe("validated_train_" + str(i), "train", item["state"], {})
    held = load("output/balance_feedback_v2/evaluation_states.pt")
    for label, state in held:
        bank["evaluation"]["balance"].append(state)
    # Reuse completed held-out measurements, never add their labels to training.
    prior = __import__("json").loads(
        Path("output/balance_feedback_v2/results.json").read_text()
    )
    records.extend(
        dict(
            label=r["reset"],
            split="evaluation",
            initial_valid=True,
            success=r["success"],
            hold=r["hold"],
            termination=r["termination"],
            reused_verified_result=True,
        )
        for r in prior["results"]
        if r["controller"] == "observable_LQR"
    )
    teacher = np.load("output/multi_impulse_preparation_v1/teacher.npz")
    world = World(assets, cfg)
    proposals = [
        (90, 0, 0, 0, 0),
        (89, 0.15, 0, 0, 0),
        (88, -0.2, 0, 0, 0),
        (85, 0.4, 0, 0, 0),
        (80, 0.6, 0, 0, 0),
        (75, 0.8, 0, 0, 0),
        (70, 1, 0, 0, 0),
        (60, 1.2, 0, 0, 0),
        (89, -0.35, 0.004, 0, 0),
        (89, -0.35, -0.004, 0.01, 0),
        (88, 0.3, 0, -0.01, 0.01),
        (85, 0.5, 0, 0, -0.02),
    ]
    for i, (pitch, omega, shift, front, rear) in enumerate(proposals):
        split = "evaluation" if i % 3 == 0 else "train"
        good, why = feasible_state(
            world,
            teacher["q"][e.qids],
            210000 + i,
            pitch=pitch,
            omega=omega,
            com_shift=shift,
            front_delta=front,
            rear_delta=rear,
        )
        details = dict(
            pitch=pitch,
            omega=omega,
            com_shift=shift,
            front_delta=front,
            rear_delta=rear,
        )
        if not good:
            records.append(
                dict(
                    label="proposal_" + str(i),
                    split=split,
                    initial_valid=False,
                    details=why,
                    **details,
                )
            )
            continue
        try:
            good, why = raw_state(
                e,
                "near_vertical",
                210000 + i,
                qpos=world.d.qpos.copy(),
                qvel=world.d.qvel.copy(),
                target=teacher["target"],
            )
        except ValueError:
            good = False
        if not good:
            records.append(
                dict(
                    label="proposal_" + str(i),
                    split=split,
                    initial_valid=False,
                    **details,
                )
            )
            continue
        probe("proposal_" + str(i), split, e.state(), details)
    # Actual saved post-takeoff states give an honest lower-angle bridge even when IK proposals fail.
    priorbank = load("output/adc_front_capture_preparation_v1/reset_bank.pt")
    for split in ["train", "evaluation"]:
        for kind in ["post_liftoff", "partial"]:
            for item in priorbank[split][kind][:3]:
                e.restore_start(item["state"], "partial")
                good, _ = validate_state(e)
                if good:
                    bank[split]["bridge"].append(item["state"])
    # ADC supervision is limited to actual valid kick-up portions of certified trajectories.
    for seed in [95010, 95011]:
        obs, _ = e.reset(1.0, seed)
        frames = []
        while True:
            with torch.no_grad():
                act = net.actor(torch.tensor(obs)).tanh().numpy()
            if e._momentum_prefix_safe and (
                not e.tracker.confirmed or e.row["pitch_velocity"] > 0.05
            ):
                frames.append((obs.copy(), act.copy()))
            (obs, _), _, done, info = e.step(act)
            if done:
                break
        if not info["certified_standing_takeoff"]:
            raise RuntimeError("ADC initialization failed takeoff certificate")
        for x, y in frames:
            obs_data.append(x)
            action_data.append(y)
            roles.append(0)
        records.append(
            dict(
                label="adc_" + str(seed),
                split="train",
                certified_takeoff=True,
                frames=len(frames),
            )
        )
    if not bank["train"]["balance"] or not bank["train"]["bridge"]:
        raise RuntimeError("Missing usable curriculum distribution")
    torch.save(bank, a.output / "bank.pt")
    np.savez_compressed(
        a.output / "demonstrations.npz",
        observations=np.stack(obs_data),
        actions=np.stack(action_data),
        roles=np.array(roles),
        centers=np.array(centers),
    )
    save_json(a.output / "capture_region.json", records)
    save_json(a.output / "config.json", vars(cfg))
    save_json(
        a.output / "manifest.json",
        {p.name: digest(p) for p in a.output.iterdir() if p.is_file()},
    )
    print(
        "Prepared",
        len(obs_data),
        "competence-certified frames; bank counts",
        [(s, k, len(v)) for s, b in bank.items() for k, v in b.items()],
        flush=True,
    )


if __name__ == "__main__":
    main()
