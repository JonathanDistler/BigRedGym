import argparse
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Deploy a policy to a Unitree Go2.")
    parser.add_argument("interface", help="Robot network interface, e.g. eth0")
    parser.add_argument(
        "--bundle",
        help="Policy bundle directory from scripts/export_go2_policy.py. "
        "Without it, the actor is built from the latest training run in logs/.",
    )
    parser.add_argument(
        "--task",
        help="Task name (e.g. go2trot); selects the CustomConfig subclass in "
        "go2_deploy/custom_implementation/. Required without --bundle; with "
        "--bundle it defaults to the bundle's task.",
    )
    parser.add_argument(
        "--command-check",
        action="store_true",
        help="Remote-activated encoder/command diagnostic; no handstand calibration",
    )
    parser.add_argument(
        "--check-joint",
        type=int,
        default=0,
        help="Canonical FL/FR/RL/RR joint index 0..11",
    )
    parser.add_argument(
        "--check-offset",
        type=float,
        default=0.0,
        help="Diagnostic excursion in radians (default hold only; max abs 0.02)",
    )
    args = parser.parse_args()
    if args.command_check and (args.bundle or args.task):
        parser.error("--command-check is separate from --bundle/--task")
    if not args.command_check and (args.check_joint != 0 or args.check_offset != 0):
        parser.error("--check-joint/--check-offset require --command-check")
    if not args.bundle and not args.task and not args.command_check:
        parser.error("--task is required when not deploying from --bundle")
    if sys.platform != "linux":
        parser.error("Go2 deployment requires Linux (timerfd).")

    try:
        from unitree_sdk2py.core import channel
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py":
            raise
        parser.error(
            "Unitree SDK is not installed. Follow README_DEPLOY.md, then run "
            "uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy "
            "<interface> --task <task>."
        )

    from go2_deploy.utility.csv_logger import new_run_dir
    from go2_deploy.utility.terminal_log import TerminalLog

    # First, so terminal.log also covers policy loading and connection errors
    run_dir = new_run_dir()
    terminal_log = TerminalLog(run_dir)
    print(f"go2_deploy {' '.join(sys.argv[1:])}")

    from go2_deploy.main_controller import MainController
    from go2_deploy.rl_controller import RLController

    # Load the policy before touching the robot, so a bad bundle fails here.
    from go2_deploy.utility.import_custom_impl import load_custom_impl

    custom_impl = load_custom_impl(args.task) if args.task else None
    if args.command_check:
        from go2_deploy.custom_implementation.go2commandcheck import make_runtime

        policy, cfg = make_runtime(args.check_joint, args.check_offset)
    elif args.bundle:
        from go2_deploy.policy_bundle import load_bundle

        # Without --task, load_bundle picks the impl for the bundle's task_name
        policy, cfg = load_bundle(args.bundle, custom_impl)
        print(f"Loaded policy bundle {args.bundle}")
    else:
        from go2_deploy.deploy_config import DeployConfig

        policy, cfg = None, DeployConfig(custom_impl)
    print(f"Using custom implementation {type(cfg.custom_impl).__name__}")
    rl_controller = RLController(cfg, policy)

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    ChannelFactoryInitialize(
        0,
        args.interface,
    )
    controller = MainController(cfg, rl_controller, run_dir=run_dir)
    terminal_log.state = lambda: controller._state.name

    # SIGTERM (the launcher's L2 + R2) and SIGHUP (tmux kill-session): run the
    # e-stop (lie down, stand back up) and exit, instead of dying mid-policy
    stop_signal = []

    def request_shutdown(signum, frame):
        stop_signal.append(signal.Signals(signum).name)
        controller.request_shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)

    print(
        "\nConnection successful! Keep the robot away from obstacles and "
        + "always have access to the remote controller\n"
    )

    while not controller.shut_down:
        time.sleep(0.2)
    terminal_log.record_exit(f"stopped ({stop_signal[0] if stop_signal else 'remote'})")


if __name__ == "__main__":
    main()
