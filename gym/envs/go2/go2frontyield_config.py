"""Opt-in bounded-residual PPO around the physically validated reference."""

import json
from pathlib import Path
from gym.envs.go2.go2kickup_config import Go2KickupCfg, Go2KickupRunnerCfg

_PARAMS = json.loads(
    (Path(__file__).parent / "references/kickup_front_yield.json").read_text()
)


class Go2FrontYieldCfg(Go2KickupCfg):
    class env(Go2KickupCfg.env):
        episode_length_s = 1.6

    class front_yield:
        batched_cpu_sync = True
        # Inline copy is persisted into effective_config.json; old runs do not
        # silently pick up future edits to the JSON reference.
        parameters = _PARAMS
        loading_front = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]
        loading_rear = [0.0, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
        push_front = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]
        push_rear = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        tuck_front = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        tuck_rear = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
        stabilize_front = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        stabilize_rear = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
        slew_rad_s = 0.5
        phase_blend_s = 0.05
        max_push_cancellation_fraction = 0.2

    class kick_motion(Go2KickupCfg.kick_motion):
        settle_seconds = _PARAMS["settle"]
        crouch_seconds = _PARAMS["crouch"]
        extend_seconds = _PARAMS["extend"] * max(
            _PARAMS["hip_duration"] + _PARAMS["hip_delay"],
            _PARAMS["knee_duration"] + _PARAMS["knee_delay"],
        )

    class kick_curriculum(Go2KickupCfg.kick_curriculum):
        clearance = [0.03] * 7
        amplitude = [1.0] * 7
        pitch_degrees = [32.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0]
        front_drift = [0.05, 0.05, 0.045, 0.045, 0.04, 0.04, 0.04]
        front_speed = [0.6] * 7

    class handstand_curriculum(Go2KickupCfg.handstand_curriculum):
        success_fraction = 0.9


class Go2FrontYieldRunnerCfg(Go2KickupRunnerCfg):
    class algorithm(Go2KickupRunnerCfg.algorithm):
        learning_rate = 1e-4

    class critic(Go2KickupRunnerCfg.critic):
        class reward:
            class weights:
                yield_takeoff = 8.0
                yield_clearance = 4.0
                yield_rotation = 2.0
                yield_terminal = 3.0
                yield_support = 1.0
                yield_instability = 1.0
                yield_drift = 1.0
                residual_motion = 0.05
                action_rate = 0.02
                torques = 5e-6
                dof_pos_limits = 1.0
                collision = 1.0

            class termination_weight:
                termination = 2.0

    class runner(Go2KickupRunnerCfg.runner):
        experiment_name = "go2frontyield"
        max_iterations = 200
        save_interval = 25
        disable_wandb = True
