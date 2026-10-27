"""Initial handstand curriculum settings; tune in simulation before deployment."""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2HandstandCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 8
        min_base_height = 0.12

    class init_state(Go2Cfg.init_state):
        reset_mode = "reset_to_handstand"

    class handstand_curriculum:
        enabled = True
        initial_level = 0
        pitch_degrees = [90.0, 90.0, 70.0, 45.0, 20.0, 0.0]
        tilt_noise_degrees = [1.0, 5.0, 5.0, 5.0, 5.0, 0.0]
        angular_velocity_noise = [0.0, 0.1, 0.1, 0.1, 0.1, 0.0]
        ground_clearance = 0.003
        replay_fraction = 0.2
        standing_fraction = 0.1
        hold_seconds = 2.0
        hold_score = 0.5
        min_episodes = 128
        success_fraction = 0.7
        required_batches = 2

    class scaling(Go2Cfg.scaling):
        # Larger residual range lets the policy reach the kick-up posture.
        dof_pos_target = 12 * [1.5]

    class reward_settings(Go2Cfg.reward_settings):
        lean_share_sigma = 0.25
        lean_gravity_x = 0.25
        support_sigma = 0.3
        min_front_load = 0.15  # fraction of randomized body weight, per foot
        kick_knee_target = -0.95
        kick_knee_sigma = 0.4
        kick_clearance = 0.12  # metres above the front feet
        balance_clearance = 0.35
        rear_load_tolerance = 0.05
        vertical_sigma = 0.25
        angular_sigma = 1.0
        linear_sigma = 0.25
        lean_threshold = 0.5
        kick_threshold = 0.5
        lean_hold_s = 0.10
        kick_hold_s = 0.04


class Go2HandstandRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = Go2RunnerCfg.actor.obs + ["handstand_stage_obs"]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["handstand_stage_obs"]

        class reward:
            class weights:
                handstand_lean = 1.0
                handstand_kick = 2.0
                handstand_balance = 8.0
                handstand_slip = 0.2
                torques = 5e-6
                action_rate = 0.1
                action_rate2 = 0.01
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 2.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2handstand"
        max_iterations = 1500
