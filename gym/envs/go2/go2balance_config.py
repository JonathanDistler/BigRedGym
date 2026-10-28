"""Top-of-handstand balance first; success unlocks controlled disturbances."""

from gym.envs.base.domain_randomization import DomainRandomizationCfg
from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg


class Go2BalanceCfg(Go2HandstandCfg):
    balance_only = True

    # Keep physics fixed while learning the nominal balance controller.
    class domain_randomization(DomainRandomizationCfg):
        pass

    class handstand_curriculum(Go2HandstandCfg.handstand_curriculum):
        pitch_degrees = [90.0] * 5
        tilt_noise_degrees = [0.0, 1.0, 3.0, 3.0, 5.0]
        angular_velocity_noise = [0.0, 0.0, 0.03, 0.03, 0.06]
        standing_fraction = 0.0
        replay_fraction = 0.2

    class balance_perturbations:
        enabled = True
        # Bounded world-XY delta velocity, m/s; these are impulses, not forces.
        max_delta_velocity = [0.0, 0.0, 0.0, 0.03, 0.06]
        interval_s = 3.0
        settled_hold_s = 0.5
        undisturbed_fraction = 0.25

    class push_robots(Go2HandstandCfg.push_robots):
        toggle = False  # use the gated balance perturbations instead

    class scaling(Go2HandstandCfg.scaling):
        dof_pos_target = 12 * [0.25]


class Go2BalanceRunnerCfg(Go2HandstandRunnerCfg):
    class actor(Go2HandstandRunnerCfg.actor):
        init_noise_std = 0.3

    class critic(Go2HandstandRunnerCfg.critic):
        class reward:
            class weights:
                # Additive shaping supplies gradients when the strict product
                # is near zero. Promotion still uses the original strict score.
                balance_orientation = 2.0
                balance_front_support = 2.0
                balance_clearance = 1.0
                balance_stability = 1.0
                handstand_balance = 4.0
                handstand_slip = 0.2
                torques = 5e-6
                action_rate = 0.1
                action_rate2 = 0.01
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 2.0

    class runner(Go2HandstandRunnerCfg.runner):
        experiment_name = "go2balance"
