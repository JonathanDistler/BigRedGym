"""Initial simulation tuning; preserve balance policy observation/action sizes."""

from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg


class Go2KickupCfg(Go2BalanceCfg):
    class kickup:
        duration_s = 0.65
        standing = [[0.0, 0.58, -1.16]] * 4
        crouched = [[0.0, 1.05, -2.1]] * 4
        extended = [[0.0, 1.05, -2.1]] * 2 + [[0.0, 0.25, -0.5]] * 2
        handoff_pitch_rad = 1.15
        handoff_max_pitch_rad = 1.75
        handoff_max_speed = 4.0

    class kickup_curriculum:
        enabled = True
        initial_level = 0
        # Intermediate goals are kicks, not balance-policy handoffs.
        pitch_degrees = [15.0, 30.0, 45.0, 60.0, 75.0]
        amplitudes = [0.35, 0.5, 0.7, 0.85, 1.0]
        min_episodes = 128
        success_fraction = 0.7
        required_batches = 2
        final_hold_s = 0.5
        residual_limit_rad = 0.15

    class balance_failure(Go2BalanceCfg.balance_failure):
        contact_grace_seconds = 0.7

    class handstand_curriculum(Go2BalanceCfg.handstand_curriculum):
        enabled = False

    class balance_perturbations(Go2BalanceCfg.balance_perturbations):
        enabled = False


class Go2KickupRunnerCfg(Go2BalanceRunnerCfg):
    class critic(Go2BalanceRunnerCfg.critic):
        class reward(Go2BalanceRunnerCfg.critic.reward):
            class weights(Go2BalanceRunnerCfg.critic.reward.weights):
                kickup_progress = 2.0

    class runner(Go2BalanceRunnerCfg.runner):
        experiment_name = "go2kickup"
