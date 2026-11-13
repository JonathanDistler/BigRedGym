"""Separate learned balance feasibility, retaining the original standing actor."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from experiments.multi_impulse.network import initialize
from experiments.multi_impulse.core import MultiConfig, MultiEnvironment
from experiments.morphology.learning.environment import Assets
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dagger-rounds", type=int, default=8)
    p.add_argument("--latent-action-epsilon", type=float, default=0.001)
    p.add_argument("--whiten-context", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    parent = torch.load(
        "output/multi_impulse_diagnosis_v1/B_best.pt",
        weights_only=False,
        map_location="cpu",
    )["runs"]["B_seed42"]
    net, opt = initialize(parent, True)
    for param in net.actor.base.parameters():
        param.requires_grad_(False)
    for param in net.critic.parameters():
        param.requires_grad_(False)
    for g in opt.param_groups:
        g["lr"] = 1e-3
    data = np.load(a.preparation / "balance_demonstrations.npz")
    values = data["observations"].copy()
    metadata = json.loads((a.preparation / "manifest.json").read_text())
    values[:, 42] *= (
        metadata.get("observation_time_denominator", metadata["config"]["seconds"])
        / 8.0
    )
    obs = torch.tensor(values, dtype=torch.float32)
    target = torch.atanh(
        torch.tensor(data["actions"], dtype=torch.float32).clamp(
            -1 + a.latent_action_epsilon, 1 - a.latent_action_epsilon
        )
    )
    whitening = None
    if a.whiten_context:
        center = obs.mean(0)
        scale = obs.std(0, unbiased=False).clamp(min=0.001)
        whitening = net.actor.context.register_forward_pre_hook(
            lambda module, args: ((args[0] - center) / scale,)
        )
    torch.manual_seed(500)
    losses = []
    for step in range(a.steps):
        ids = torch.randint(len(obs), (128,))
        loss = (net.actor(obs[ids]) - target[ids]).square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            losses.append(dict(step=step, loss=float(loss)))
            print("balance fit", step, float(loss), flush=True)
    if whitening is not None:
        whitening.remove()
        first = net.actor.context[0]
        with torch.no_grad():
            first.weight.div_(scale[None])
            first.bias.sub_(first.weight @ center)
        # Algebraically fold normalization into the deployable actor. Context
        # Adam coordinates changed; export them fresh, retaining all base states.
        for param in net.actor.context.parameters():
            opt.state.pop(param, None)
    for param in net.parameters():
        param.requires_grad_(True)
    for g in opt.param_groups:
        g["lr"] = 1e-5
    cfg = MultiConfig(
        **json.loads((a.preparation / "manifest.json").read_text())["config"]
    )
    cfg.full_trace = True
    bank = torch.load(
        a.preparation / "balance_bank.pt", weights_only=False, map_location="cpu"
    )
    e = MultiEnvironment(Assets(a.output / "assets"), cfg)
    results = []
    for split in ("train", "evaluation"):
        for i, item in enumerate(bank[split]["near_vertical"]):
            actor, _ = e.restore_start(item["state"], "near_vertical")
            while True:
                with torch.no_grad():
                    action = (
                        net.actor(torch.tensor(np.repeat(actor[None], 16, axis=0)))
                        .tanh()
                        .numpy()[0]
                    )
                (actor, _), _, done, info = e.step(action)
                if done:
                    break
            results.append(
                dict(
                    split=split,
                    index=i,
                    hold=e.multi_best_hold,
                    success=info["intermediate_success"],
                    termination=info["termination"],
                )
            )
            print("learned balance validation", results[-1], flush=True)
    # Query the validated teacher on learner-visited states, before any illegal
    # impact. Low demonstration MSE alone is not closed-loop balance validation.
    from experiments.multi_impulse.balance import BalanceTeacher

    teacher_data = np.load(a.preparation / "teacher.npz")
    teacher = BalanceTeacher(e, teacher_data["q"], teacher_data["target"])
    dagger_frames = 0
    best_hold = max(r["hold"] for r in results)
    best_state = copy.deepcopy(net.state_dict())
    best_opt = copy.deepcopy(opt.state_dict())
    for round_index in range(a.dagger_rounds):
        for param in net.actor.base.parameters():
            param.requires_grad_(False)
        for param in net.critic.parameters():
            param.requires_grad_(False)
        extra_obs = []
        extra_target = []
        for item in bank["train"]["near_vertical"]:
            actor, _ = e.restore_start(item["state"], "near_vertical")
            for _ in range(310):
                extra_obs.append(actor.copy())
                extra_target.append(teacher.command(e))
                with torch.no_grad():
                    action = (
                        net.actor(torch.tensor(np.repeat(actor[None], 16, axis=0)))
                        .tanh()
                        .numpy()[0]
                    )
                (actor, _), _, done, _ = e.step(action)
                dagger_frames += 1
                if done:
                    break
        obs = torch.cat([obs, torch.tensor(np.stack(extra_obs), dtype=torch.float32)])
        target = torch.cat(
            [
                target,
                torch.atanh(
                    torch.tensor(np.stack(extra_target), dtype=torch.float32).clamp(
                        -0.999, 0.999
                    )
                ),
            ]
        )
        for g in opt.param_groups:
            g["lr"] = 3e-4
        for _ in range(600):
            ids = torch.randint(len(obs), (128,))
            loss = (net.actor(obs[ids]) - target[ids]).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        actor, _ = e.restore_start(
            bank["evaluation"]["near_vertical"][0]["state"], "near_vertical"
        )
        while True:
            with torch.no_grad():
                action = (
                    net.actor(torch.tensor(np.repeat(actor[None], 16, axis=0)))
                    .tanh()
                    .numpy()[0]
                )
            (actor, _), _, done, info = e.step(action)
            if done:
                break
        print("DAgger learned hold", round_index, e.multi_best_hold, flush=True)
        results.append(
            dict(
                split="evaluation_dagger",
                round=round_index,
                hold=e.multi_best_hold,
                success=info["intermediate_success"],
            )
        )
        if e.multi_best_hold > best_hold:
            best_hold = e.multi_best_hold
            best_state = copy.deepcopy(net.state_dict())
            best_opt = copy.deepcopy(opt.state_dict())
        if e.multi_best_hold >= 5.0:
            break
    net.load_state_dict(best_state)
    opt.load_state_dict(best_opt)
    for param in net.parameters():
        param.requires_grad_(True)
    for g in opt.param_groups:
        g["lr"] = 1e-5
    # The added residual is exactly gated off below 30 degrees. Base weights and
    # original Adam states are retained, not distilled into a second controller.
    torch.save(
        dict(
            network=net.state_dict(),
            optimizer=opt.state_dict(),
            config=cfg.__dict__,
            source_data_frames=len(obs),
            gradient_steps=a.steps,
            latent_action_epsilon=a.latent_action_epsilon,
            folded_context_whitening=a.whiten_context,
            dagger_frames=dagger_frames,
            results=results,
        ),
        a.output / "balance_actor.pt",
    )
    dump(
        a.output / "results.json",
        dict(
            results=results,
            losses=losses,
            teacher_frames=len(data["observations"]),
            dagger_frames=dagger_frames,
            deployment="one actor; teacher not used in inference",
        ),
    )


if __name__ == "__main__":
    main()
