"""Time-major vector PPO collection and separate standing/intermediate evaluation."""

import numpy as np
import torch
from experiments.morphology.learning.ppo import advantages
from experiments.front_catch.vector import KINDS


def collect(net, vector, mode, actor, critic, horizon, mixture, hp):
    observations = []
    privileged = []
    raws = []
    logs = []
    values = []
    rewards = []
    dones = []
    nextvalues = []
    episodes = []
    samples = []
    for _ in range(horizon):
        with torch.no_grad():
            raw, log, value = net.sample(torch.tensor(actor), torch.tensor(critic))
        results = vector.step(mode, raw.tanh().numpy(), mixture)
        observations.append(actor.copy())
        privileged.append(critic.copy())
        raws.append(raw.numpy())
        logs.append(log.numpy())
        values.append(value.numpy())
        actor = np.stack([r[0] for r in results])
        critic = np.stack([r[1] for r in results])
        terminal = np.stack([r[4] for r in results])
        with torch.no_grad():
            nv = net.critic(torch.tensor(terminal)).squeeze(-1).numpy()
        rewards.append([r[2] for r in results])
        dones.append([r[3] for r in results])
        nextvalues.append(nv)
        episodes.extend(r[5] for r in results if r[5] is not None)
        samples.extend(r[6] for r in results)
    a, returns = advantages(
        np.array(rewards),
        np.array(values),
        np.array(nextvalues),
        np.array(dones),
        hp["gamma"],
        hp["gae_lambda"],
    )

    def flat(items):
        x = np.asarray(items, dtype=np.float32)
        return torch.tensor(x.reshape((-1,) + x.shape[2:]))

    batch = (
        flat(observations),
        flat(privileged),
        flat(raws),
        flat(logs),
        flat(returns),
        flat(a),
    )
    return batch, actor, critic, episodes, samples


def hp_defaults():
    return dict(
        gamma=0.995,
        gae_lambda=0.95,
        epochs=2,
        batch_size=256,
        learning_rate=1e-5,
        clip=0.2,
        target_kl=0.01,
        entropy_weight=0.005,
        value_weight=0.5,
        max_grad_norm=1.0,
        bc_weight=0.02,
        bc_latent_scale=0.05,
        bc_batch_size=128,
        bc_phase_fraction=1.0,
    )


def evaluate(
    net, vector, mode, condition, bank, standing_seeds=(96000, 96001, 96002, 96003)
):
    specs = [dict(kind="standing", seed=seed) for seed in standing_seeds]
    # One independently held-out state of each type, never mixed into standing scores.
    specs.extend(
        dict(kind=kind, index=0, seed=97000 + i) for i, kind in enumerate(KINDS[1:])
    )
    if len(specs) > vector.n:
        raise ValueError("Evaluation needs at least nine vector slots")
    specs += [None] * (vector.n - len(specs))
    actor, critic = vector.evaluation_reset(mode, condition, specs)
    active = np.array([s is not None for s in specs])
    rows = []
    traces = []
    steps = 0
    for _ in range(410):
        if not active.any():
            break
        with torch.no_grad():
            actions = net.actor(torch.tensor(actor)).tanh().numpy()
        results = vector.evaluation_step(mode, actions, active)
        steps += int(active.sum())
        for i, (result, spec) in enumerate(zip(results, specs, strict=True)):
            if not active[i]:
                continue
            traces.append(dict(**result[6], evaluation_seed=spec["seed"], world=i))
            if result[3]:
                rows.append(
                    dict(
                        **result[5],
                        evaluation_seed=spec["seed"],
                        evaluation_type="standing"
                        if spec["kind"] == "standing"
                        else "intermediate",
                    )
                )
                active[i] = False
        actor = np.stack([r[0] for r in results])
    if active.any():
        raise RuntimeError("Evaluation exceeded finite episode horizon")
    return rows, traces, steps
