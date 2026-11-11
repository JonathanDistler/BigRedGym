"""Opt-in six-stage recovery curriculum; no existing task defaults change."""

from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg


class Go2KickupRecoveryCfg(Go2BalanceCfg):
    class contact_cpg(Go2BalanceCfg.contact_cpg):
        enabled = True
        modes = ["stance", "stance", "swing", "swing"]

    class recovery:
        dataset = "datasets/go2kickup_recovery_train_v2/train.npz"

    class env(Go2BalanceCfg.env):
        num_envs = 16
        episode_length_s = 8

    class handstand_curriculum(Go2BalanceCfg.handstand_curriculum):
        pitch_degrees = [90.0] * 6
        tilt_noise_degrees = [0.0, 2.0, 8.0, 0.0, 0.0, 0.0]
        angular_velocity_noise = [0.0, 0.12, 0.3, 0.0, 0.0, 0.0]
        min_episodes = 128
        required_batches = 2
        success_fraction = 0.8
        hold_seconds = 5.0

    class balance_perturbations(Go2BalanceCfg.balance_perturbations):
        enabled = False
        max_delta_velocity = [0.0] * 6


class Go2KickupRecoveryRunnerCfg(Go2BalanceRunnerCfg):
    class actor(Go2BalanceRunnerCfg.actor):
        obs = Go2BalanceRunnerCfg.actor.obs + ["cpg_phase_obs", "cpg_contact_obs"]

    class critic(Go2BalanceRunnerCfg.critic):
        obs = Go2BalanceRunnerCfg.critic.obs + ["cpg_phase_obs", "cpg_contact_obs"]

        class reward(Go2BalanceRunnerCfg.critic.reward):
            class weights(Go2BalanceRunnerCfg.critic.reward.weights):
                recovery_com_velocity = 3.0
                recovery_com_offset = 3.0
                recovery_hold = 20.0

    class algorithm(Go2BalanceRunnerCfg.algorithm):
        rollout_size = 4096
        batch_size = 1024

    class runner(Go2BalanceRunnerCfg.runner):
        experiment_name = "go2kickup_recovery"
        max_iterations = 200
        save_interval = 25
        disable_wandb = True
