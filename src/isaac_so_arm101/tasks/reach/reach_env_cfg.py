# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.classic.cartpole.mdp.rewards import joint_pos_target_l2 # added this to use joint_pos_target_l2

from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
import torch
from isaaclab.utils.math import quat_from_euler_xyz


##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # # world
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    # )
    
    # Load your custom background scene (replaces ground and table)
    # Use the {ENV_REGEX_NS} to ensure the scene is cloned for every environment instance
    custom_env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/cirp-lab/moore/palm_tree_models/blender/pretoria_gardens_4k/pretoria_gardens_4k_env_v2.usdc",
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.25, -0.1),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.0), # reduced the upper bound from 1.5 to stay within limits
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # increased from -0.0001
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # # added this wrist flex reward
    # # Checking if removing the wrist flex tracking helps with movement.
    # wrist_flex_tracking = RewTerm(
    #     func=joint_pos_target_l2,
    #     weight=-0.05,  # tune this relative to position reward. make sure this is less than end_effector_position_tracking, else the policy will learn to hold the wrist at 'target' while ignoring where the arm is in space.
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["wrist_pitch"]),
    #         "target": 1.5708,  # target angle in radians — set your desired angle here
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.0005, "num_steps": 200000} # changed weight from -0.005 and num_steps from 4500
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.0002, "num_steps": 200000} # changed weight from -0.001 and num_steps from 4500
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    # scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=50.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

##
# VLA Specific Configuration
##

# 1. Convert your Euler angles (degrees) to a Quaternion (w, x, y, z)
# Euler: X=-2.0, Y=-9.067, Z=-90.0
camera_quat = quat_from_euler_xyz(
    torch.tensor([-2.0 * (3.14159 / 180.0)]), 
    torch.tensor([-9.067 * (3.14159 / 180.0)]), 
    torch.tensor([-90.0 * (3.14159 / 180.0)])
)

@configclass
class ReachVlaSceneCfg(ReachSceneCfg):
    """Extends the scene to spawn a camera on every robot instance."""
    
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        # The prim_path tells Isaac Lab to put the camera under the gripper link of every robot
        prim_path="{ENV_REGEX_NS}/Robot/sts3215_gripper/camera",
        update_period=0.0, # Updates every simulation step
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        width=256,
        height=256,
        # Apply your exact X, Y, Z position and the converted Rotation
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.01371, 0.03346, -0.02114),
            rot=(camera_quat[0, 0].item(), camera_quat[0, 1].item(), camera_quat[0, 2].item(), camera_quat[0, 3].item()),
            convention="opengl", # Standard Isaac Sim convention
        ),
    )

@configclass
class ReachVlaEnvCfg(ReachEnvCfg):
    """The main VLA environment config that inherits from your original reach config."""
    def __post_init__(self):
        super().__post_init__()

        # 1. Swap the scene to our new VLA-enabled scene
        self.scene = ReachVlaSceneCfg(num_envs=self.scene.num_envs, env_spacing=self.scene.env_spacing)

        # 2. Setup the Arm Action (5-DoF IK)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", 
            joint_names=["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"],
            body_name="sts3215_gripper", # EXACT match from URDF for the 5th joint's child link
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=True, 
                ik_method="dls"         
            ),
        )

        # 3. Setup the Gripper Action (1-DoF Direct Position)
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_moving"], # EXACT match from URDF
            scale=1.0, 
            use_default_offset=False 
        )
        
# can add a ReachEnvCfg_PLAY config to customize the environment during playback