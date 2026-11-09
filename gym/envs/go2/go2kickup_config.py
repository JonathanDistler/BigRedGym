"""Kick-up residual policy; seven success-gated stages end at standing starts."""

from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg


class Go2KickupCfg(Go2HandstandCfg):
    class handstand_curriculum(Go2HandstandCfg.handstand_curriculum):
        pitch_degrees = [90.0, 80.0, 65.0, 50.0, 35.0, 15.0, 0.0]
        tilt_noise_degrees = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 0.0]
        angular_velocity_noise = [0.0, 0.02, 0.03, 0.04, 0.05, 0.05, 0.0]
        hold_seconds = 1.0

    class kick_motion:
        crouch_seconds = 0.35
        extend_seconds = 0.25
        front_thigh = 1.1
        front_knee = -2.2
        rear_thigh = 1.25
        rear_knee = -2.5
        rear_extend_thigh = 0.5
        rear_extend_knee = -1.0

    class reward_settings(Go2HandstandCfg.reward_settings):
        valid_front_speed = 0.10
        max_front_displacement = 0.10
        min_rear_clearance = 0.12
        motion_sigma = 0.5

    class contact_cpg(Go2HandstandCfg.contact_cpg):
        # Contact timing can be enabled when composing with the front CPG policy.
        modes = ["stance", "stance", "ignore", "ignore"]


class Go2KickupRunnerCfg(Go2HandstandRunnerCfg):
    class actor(Go2HandstandRunnerCfg.actor):
        obs = Go2HandstandRunnerCfg.actor.obs + ["kick_motion_obs"]

    class critic(Go2HandstandRunnerCfg.critic):
        obs = Go2HandstandRunnerCfg.critic.obs + ["kick_motion_obs"]

        class reward:
            class weights:
                handstand_lean = 1.0
                kick_motion = 0.5
                handstand_kick = 3.0
                kick_clearance = 2.0
                handstand_balance = 10.0
                handstand_slip = 10.0
                kick_front_displacement = 10.0
                torques = 5e-6
                action_rate = 0.1
                action_rate2 = 0.01
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 5.0

    class runner(Go2HandstandRunnerCfg.runner):
        experiment_name = "go2kickup"
