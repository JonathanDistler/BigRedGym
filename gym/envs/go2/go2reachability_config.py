"""Expand a front-supported recovery distribution outward from the goal."""

from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg


class Go2ReachabilityCfg(Go2BalanceCfg):
    class env(Go2BalanceCfg.env):
        episode_length_s = 8

    class balance_failure(Go2BalanceCfg.balance_failure):
        # Allow a short recovery before treating loss of both front feet as a fall.
        both_front_airborne_seconds = 0.25

    class reward_settings(Go2BalanceCfg.reward_settings):
        # Contact shaping targets meaningful weight bearing, independently of
        # the 1 N threshold used to detect an airborne foot.
        front_support_fraction = 0.25  # body weight per front foot
        recovery_angular_sigma = 2.0  # rad/s; leave room for corrective rotation

    class reachability:
        # Fractions of joint travel and torque-work speed envelope, respectively.
        position_fraction = [0.0, 0.025, 0.05, 0.10, 0.15, 0.25]
        velocity_fraction = [0.0, 0.0, 0.01, 0.025, 0.05, 0.10]
        attempts = 24
        probe_seconds = 0.04
        # Screening limits apply to reset proposals, not live termination.
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
    class critic(Go2BalanceRunnerCfg.critic):
        class reward(Go2BalanceRunnerCfg.critic.reward):
            class weights(Go2BalanceRunnerCfg.critic.reward.weights):
                front_contact_loss = 1.0
                reachability_stability = 0.25

    class runner(Go2BalanceRunnerCfg.runner):
        experiment_name = "go2reachability"
