"""Standing-start rear kick curriculum, independent of handstand balancing."""

from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg


class Go2KickupCfg(Go2HandstandCfg):
    class env(Go2HandstandCfg.env):
        num_envs = 16
        episode_length_s = 4.0

    class handstand_curriculum(Go2HandstandCfg.handstand_curriculum):
        # Retain inherited persistence and completed-episode batch accounting.
        pitch_degrees = [0.0] * 7
        tilt_noise_degrees = [0.0] * 7
        angular_velocity_noise = [0.0] * 7
        replay_fraction = 0.0
        standing_fraction = 0.0
        ground_clearance = 0.001
        hold_seconds = 0.06
        success_fraction = 0.4
        min_episodes = 64
        required_batches = 2

    class kick_motion:
        settle_seconds = 0.15
        crouch_seconds = 0.55
        extend_seconds = 0.30
        rear_thigh = 1.15
        rear_knee = -2.3
        rear_extend_thigh = 0.30
        rear_extend_knee = -0.85

    class kick_curriculum:
        clearance = [0.005, 0.012, 0.025, 0.045, 0.07, 0.11, 0.16]
        amplitude = [0.12, 0.20, 0.32, 0.46, 0.62, 0.80, 1.0]
        pitch_degrees = [2.0, 4.0, 7.0, 11.0, 16.0, 23.0, 32.0]
        front_drift = [0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025]
        front_speed = [0.30, 0.25, 0.22, 0.18, 0.15, 0.12, 0.10]
        support_grace_s = 0.08

    class scaling(Go2HandstandCfg.scaling):
        dof_pos_target = [0.08, 0.12, 0.12] * 2 + [0.25, 0.45, 0.45] * 2

    class reward_settings(Go2HandstandCfg.reward_settings):
        min_front_load = 0.06
        motion_sigma = 0.5


class Go2KickupRunnerCfg(Go2HandstandRunnerCfg):
    class actor(Go2HandstandRunnerCfg.actor):
        obs = [
            x for x in Go2HandstandRunnerCfg.actor.obs if x != "handstand_stage_obs"
        ] + ["kick_motion_obs", "kick_goal_obs", "kick_support_obs"]
        init_noise_std = 0.3

    class critic(Go2HandstandRunnerCfg.critic):
        obs = [
            x for x in Go2HandstandRunnerCfg.critic.obs if x != "handstand_stage_obs"
        ] + ["kick_motion_obs", "kick_goal_obs", "kick_support_obs"]

        class reward:
            class weights:
                kick_clearance = 4.0
                kick_front_support = 0.5
                kick_pitch = 0.5
                kick_motion = 0.15
                handstand_slip = 4.0
                kick_front_displacement = 10.0
                kick_instability = 0.05
                residual_motion = 0.1
                torques = 5e-6
                action_rate = 0.05
                action_rate2 = 0.005
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 0.5

    class algorithm(Go2HandstandRunnerCfg.algorithm):
        rollout_size = 8192
        batch_size = 1024
        max_gradient_steps = 16
        learning_rate = 3e-4

    class runner(Go2HandstandRunnerCfg.runner):
        experiment_name = "go2kickup"
        save_interval = 25
        resume = False
