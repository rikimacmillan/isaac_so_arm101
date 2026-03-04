from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration
##

PING_TI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/PingTi_Arm_v3.urdf",
        activate_contact_sensors=False, # set as false while waiting for capsule implementation
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        # Exact joint names extracted
        joint_pos={
            "base_yaw": 0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper_moving": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.9,
    
        # "base": ImplicitActuatorCfg(
        #     joint_names_expr=["base_yaw"],
        #     effort_limit_sim=2.94,     # 1x ST3215
        #     velocity_limit_sim=4.76,
        #     stiffness=200.0,
        #     damping=80.0,
        # ),
        # "shoulder": ImplicitActuatorCfg(
        #     joint_names_expr=["shoulder_pitch"],
        #     effort_limit_sim=9.80,     # 2x STS3250
        #     velocity_limit_sim=8.05,
        #     stiffness=400.0,
        #     damping=160.0,
        # ),
        # "elbow": ImplicitActuatorCfg(
        #     joint_names_expr=["elbow_pitch"],
        #     effort_limit_sim=5.88,     # 2x ST3215
        #     velocity_limit_sim=4.76,
        #     stiffness=240.0,
        #     damping=90.0,
        # ),
        # "wrist": ImplicitActuatorCfg(
        #     joint_names_expr=["wrist_pitch", "wrist_roll"],
        #     effort_limit_sim=2.94,     # 1x ST3215 per axis
        #     velocity_limit_sim=4.76,
        #     stiffness={
        #         "wrist_pitch": 80.0,
        #         "wrist_roll": 50.0,
        #     },
        #     damping={
        #         "wrist_pitch": 30.0,
        #         "wrist_roll": 20.0,
        #     },
        # ),
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["gripper_moving"],
        #     effort_limit_sim=2.94,     # 1x ST3215
        #     velocity_limit_sim=4.76,
        #     stiffness=60.0,
        #     damping=20.0,
        # ),
)

