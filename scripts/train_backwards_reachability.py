"""Train recovery, optionally initializing networks from a successful balance policy."""

import argparse

from scripts.train import get_train_args, setup, train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--warm_start", help="Balance checkpoint; fresh optimizers and curriculum"
    )
    options, remaining = parser.parse_known_args()
    args = get_train_args(["--task", "go2reachability", *remaining])
    if options.warm_start and args.resume:
        parser.error("choose warm_start or resume")
    train_cfg, runner = setup(args)
    if options.warm_start:
        import torch

        saved = torch.load(
            options.warm_start, weights_only=True, map_location=args.device
        )
        runner.alg.actor.load_state_dict(saved["actor_state_dict"])
        runner.alg.critic.load_state_dict(saved["critic_state_dict"])
    train(train_cfg, runner)


if __name__ == "__main__":
    main()
