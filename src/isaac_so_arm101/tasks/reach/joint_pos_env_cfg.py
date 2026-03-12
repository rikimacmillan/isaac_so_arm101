# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.reach.reach_env_cfg import ReachEnvCfg
from isaac_so_arm101.robots.pingti.pingti import PING_TI_CFG # included the pingti config

##
# Scene definition
##


@configclass
class SoArm100ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper"]

        # TODO: reorient command target

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachEnvCfg_PLAY(SoArm100ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm101ReachEnvCfg_PLAY(SoArm101ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

@configclass
class PingTiReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = PING_TI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["moving_gripper"] # link name used for reward
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["moving_gripper"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["moving_gripper"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"], # maybe exclude the actual moving gripper for reaching tasks?
            scale=0.1, # reduced from 0.5 for smaller steps, maybe more stable
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["moving_gripper"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        
        # Override target ranges for PingTi — narrow upward target
        self.commands.ee_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),   # directly above base
            pos_y=(-0.05, 0.05),   # directly above base
            pos_z=(0.6, 0.7),      # 60-70cm above base = straight up
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )

@configclass
class PingTiReachEnvCfg_PLAY(PingTiReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
