"""Train recovery, optionally initializing networks from a successful balance policy."""

import sys

from scripts.train import get_train_args, setup, train


def main():
    args = get_train_args(["--task", "go2reachability", *sys.argv[1:]])
    train_cfg, runner = setup(args)
    train(train_cfg, runner)


if __name__ == "__main__":
    main()
