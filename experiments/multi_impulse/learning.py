"""KL-validated PPO steps; rejected attempts consume no extra environment data."""

import copy
import torch
from experiments.morphology.learning.ppo import update


def safe_update(net, opt, batch, hp, demos):
    weights = copy.deepcopy(net.state_dict())
    state = copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state()
    for retry in range(8):
        if retry:
            net.load_state_dict(weights)
            opt.load_state_dict(copy.deepcopy(state))
            torch.set_rng_state(rng)
        for group in opt.param_groups:
            group["lr"] = state["param_groups"][0]["lr"] * (0.5**retry)
        stats = update(net, opt, batch, hp, demos)
        if stats["post_update_kl"] <= 1.5 * hp["target_kl"]:
            stats.update(
                learning_rate=opt.param_groups[0]["lr"],
                kl_backtracks=retry,
                actual_latent_std=float(
                    net.distribution(batch[0]).stddev.detach().mean()
                ),
            )
            return stats
    net.load_state_dict(weights)
    opt.load_state_dict(copy.deepcopy(state))
    torch.set_rng_state(rng)
    raise RuntimeError(
        "No numerically stable KL-bounded PPO step; checkpoint preserved"
    )
