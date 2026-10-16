"""Measure a legged policy's height and falls without opening a viewer.

Uses the same checkpoint/config selection as play.py. A low posture is
reported separately because base-contact termination can miss crawling.
"""

import sys

import torch

from scripts.play import get_play_args, setup


def evaluate(env, runner, seconds=5.0, height=None):
    initial_height = env.root_states[:, 2].clone()
    minimum_height = initial_height.clone()
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    low = torch.zeros_like(fell)
    height_sum = torch.zeros_like(initial_height)
    error_sum = torch.zeros_like(initial_height)
    slip_sum = torch.zeros_like(initial_height)
    initial_xy = env.root_states[:, :2].clone()
    samples = 0
    steps = int(seconds / env.dt)
    for step in range(steps):
        if height is not None:
            env.commands[:, :3] = 0
            env.commands[:, 3] = height
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        minimum_height = torch.minimum(minimum_height, env.root_states[:, 2])
        fell |= env.terminated
        low |= env.root_states[:, 2] < 0.20
        if height is not None and step >= steps // 2:
            height_sum += env.root_states[:, 2]
            error_sum += (env.root_states[:, 2] - env.commands[:, 3]).abs()
            contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
            foot_speed = env._rigid_body_lin_vel[:, env.feet_indices, :2].norm(dim=2)
            slip_sum += (foot_speed * contact).sum(dim=1) / contact.sum(dim=1).clamp(
                min=1
            )
            samples += 1
    print(f"Evaluation: {seconds:g}s, {env.num_envs} robots, no episode resets")
    print(
        "Fall terminations (contact or configured low height): "
        f"{fell.sum().item()}/{env.num_envs}"
    )
    print(f"Robots below 0.20m: {low.sum().item()}/{env.num_envs}")
    print(f"Mean minimum base height: {minimum_height.mean().item():.3f}m")
    print(f"Mean final base height: {env.root_states[:, 2].mean().item():.3f}m")
    if height is not None:
        print(
            f"Target: {height:.3f}m; settled mean height: "
            f"{height_sum.mean().item() / samples:.3f}m; "
            f"mean absolute error: {error_sum.mean().item() / samples:.3f}m"
        )
        drift = (env.root_states[:, :2] - initial_xy).norm(dim=1).mean().item()
        print(
            f"Mean base drift: {drift:.3f}m; "
            f"settled contact-foot speed: {slip_sum.mean().item() / samples:.3f}m/s"
        )


if __name__ == "__main__":
    argv = sys.argv[1:]
    height_sweep = "--height_sweep" in argv
    if height_sweep:
        argv.remove("--height_sweep")
    args = get_play_args(argv)
    args.headless = True
    if args.seed is None:
        args.seed = 42
    torch.set_num_threads(1)
    with torch.no_grad():
        env, runner = setup(args)
        try:
            if not hasattr(env, "root_states"):
                raise ValueError("Evaluation requires a legged robot task")
            if height_sweep:
                lower, upper = env.command_ranges["base_height"]
                for height in (lower, (lower + upper) / 2, upper):
                    env.reset()
                    evaluate(env, runner, height=height)
            else:
                evaluate(env, runner)
        finally:
            env._backend.close()
