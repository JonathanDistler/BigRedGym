from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotRunnerCfg,
)

# Read the physical leg dimensions from the robot used by both backends.
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from gym import GYM_ROOT_DIR

_urdf = ET.parse(Path(GYM_ROOT_DIR) / "resources/robots/go2/urdf/go2.urdf")
LEG_LENGTH = abs(
    float(_urdf.find(".//joint[@name='FL_calf_joint']/origin").get("xyz").split()[2])
)
FOOT_RADIUS = float(
    _urdf.find(".//link[@name='FL_foot']/collision/geometry/sphere").get("radius")
)
STANDING_THIGH = 0.58
BASE_HEIGHT_REF = 2 * LEG_LENGTH * math.cos(STANDING_THIGH) + FOOT_RADIUS

GO2_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

GO2_FOOT_NAMES = [
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]


class Go2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2**12
        num_actuators = 12
        episode_length_s = 3
        # A collapsed robot can rest on its legs without torso contact.
        # Keep this below the lowest requested standing height (0.30 m).
        min_base_height = 0.20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "hip": 0.0,
            "thigh": STANDING_THIGH,
            "calf": -2 * STANDING_THIGH,
        }

        # * reset setup chooses how the initial conditions are chosen.
        # * "reset_to_basic" = a single position
        # * "reset_to_range" = uniformly random from a range defined below
        reset_mode = "reset_to_range"

        # * default COM for basic initialization
        pos = [0.0, 0.0, BASE_HEIGHT_REF + 0.005]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initialization for random range setup
        dof_pos_range = {
            "hip": [-0.01, 0.01],
            "thigh": [STANDING_THIGH - 0.01, STANDING_THIGH + 0.01],
            "calf": [-2 * STANDING_THIGH - 0.01, -2 * STANDING_THIGH + 0.01],
        }
        dof_vel_range = {"hip": [0.0, 0.0], "thigh": [0.0, 0.0], "calf": [0.0, 0.0]}
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [BASE_HEIGHT_REF + 0.005, BASE_HEIGHT_REF + 0.005],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.0, 0.0],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(LeggedRobotCfg.control):
        # * PD Drive parameters:
        stiffness = {"hip": 60.0, "thigh": 60.0, "calf": 60.0}
        damping = {"hip": 3.0, "thigh": 3.0, "calf": 3.0}
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class commands:
        # * time before command are changed[s]
        resampling_time = 1.0
        standing_probability = 0.5

        class ranges:
            lin_vel_x = [-2.0, 3.0]  # min max [m/s]
            lin_vel_y = 1.0  # max [m/s]
            yaw_vel = 3  # max [rad/s]

            # Leave knee flexion at the upper target rather than demanding
            # nearly straight legs (two 0.213 m links plus the foot radius).
            base_height = [0.28, 0.41]

    class push_robots:
        toggle = False
        interval_s = 1
        max_push_vel_xy = 0.5
        push_box_dims = [0.3, 0.1, 0.1]  # x,y,z [m]

    class domain_randomization(LeggedRobotCfg.domain_randomization):
        class startup(LeggedRobotCfg.domain_randomization.startup):
            contact_friction_range = [0.5, 1.0]
            link_mass_scale_range = [0.9, 1.1]

        class episode(LeggedRobotCfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.9, 1.1],
                "d_gains": [0.9, 1.1],
            }

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/" + "go2/urdf/go2.urdf"
        # Use the SDK's OBJ visuals; keep our URDF's complete physical model.
        vsim_visual_mesh_dir = "{GYM_ROOT_DIR}/thirdparty/vlearn/assets/go2/assets"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["foot"]
        fix_base_link = False
        disable_gravity = False
        disable_motors = False
        joint_damping = 0.01
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4

        class robot_layout:
            version = "go2_v1"
            dof_names = GO2_DOF_NAMES
            actuated_dof_names = GO2_DOF_NAMES
            body_groups = {"feet": GO2_FOOT_NAMES}

    class reward_settings(LeggedRobotCfg.reward_settings):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.0
        base_height_target = BASE_HEIGHT_REF
        tracking_sigma = 0.25

    class scaling(LeggedRobotCfg.scaling):
        base_ang_vel = 0.3
        base_lin_vel = 3.0
        dof_vel = 4 * [2.0, 2.0, 4.0]
        base_height = 0.3
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        dof_pos_obs = dof_pos
        dof_pos_target = 12 * [0.25]
        tau_ff = 4 * [23.7, 23.7, 45.43]
        # Observations divide by these values; scaling does not clip commands.
        commands = [3.0, 1.0, 3.0, 0.40]


class Go2RunnerCfg(LeggedRobotRunnerCfg):
    seed = -1
    runner_class_name = "OnPolicyRunner"

    class actor(LeggedRobotRunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False
        actions = ["dof_pos_target"]
        add_noise = False
        disable_actions = False

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

    class critic(LeggedRobotRunnerCfg.critic):
        hidden_dims = [128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]

        class reward:
            class weights:
                tracking_lin_vel = 4.0
                tracking_ang_vel = 2.0
                lin_vel_z = 0.0
                ang_vel_xy = 0.5
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                # Command tracking replaces the fixed-height objective so
                # crouching is not penalized for being below 0.40m.
                min_base_height = 0.0
                action_rate = 1.0
                action_rate2 = 0.1
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = .2 #had originally been 0
                dof_near_home = 0.0
                tracking_base_height = 4.0 #had previously been 10 
                unwanted_motion = 2.0
                residual_motion = 1 #had previously been .5

            class termination_weight:
                termination = 2.0

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        # Preserve the old runner's collection geometry: it collected one
        # optimizer batch per rollout before these sizes became independent.
        rollout_size = 2**15

    class runner(LeggedRobotRunnerCfg.runner):
        run_name = ""
        experiment_name = "go2"
        max_iterations = 500
        algorithm_class_name = "PPO2"
