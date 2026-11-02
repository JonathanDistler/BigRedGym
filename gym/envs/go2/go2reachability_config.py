"""Expand a front-supported recovery distribution outward from the goal."""

from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg


class Go2ReachabilityCfg(Go2BalanceCfg):
    class env(Go2BalanceCfg.env):
        episode_length_s = 8

    class reachability:
        # Fractions of joint travel and torque-work speed envelope, respectively.
        position_fraction = [0.0, 0.025, 0.05, 0.10, 0.15, 0.25]
        velocity_fraction = [0.0, 0.0, 0.01, 0.025, 0.05, 0.10]
        attempts = 24
        probe_seconds = 0.04
        contact_grace_seconds = 0.03
        min_normal_force = 1.0
        max_slip_speed = 0.15

    class handstand_curriculum(Go2BalanceCfg.handstand_curriculum):
        hold_seconds = 2.0
        success_fraction = 0.7
        pitch_degrees = [90.0] * 6
        tilt_noise_degrees = [0.0] * 6
        angular_velocity_noise = [0.0] * 6
        ground_clearance = -0.0005

    class balance_perturbations(Go2BalanceCfg.balance_perturbations):
        enabled = False
        max_delta_velocity = [0.0] * 6


class Go2ReachabilityRunnerCfg(Go2BalanceRunnerCfg):
    class runner(Go2BalanceRunnerCfg.runner):
        experiment_name = "go2reachability"
