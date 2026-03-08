from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration
##

# Motor specs reference:
#   STS3215: stall torque = 2.94 Nm, no-load speed = 4.76 rad/s (1x)
#   STS3250: stall torque = 4.90 Nm, no-load speed = 8.05 rad/s (1x)
#
# Joint -> motor mapping (from hardware):
#   base_yaw      : 1x STS3215 -> effort=2.94,  velocity=4.76
#   shoulder_pitch: 2x STS3250 -> effort=9.80,  velocity=8.05
#   elbow_pitch   : 2x STS3215 -> effort=5.88,  velocity=4.76
#   wrist_pitch   : 1x STS3215 -> effort=2.94,  velocity=4.76
#   wrist_roll    : 1x STS3215 -> effort=2.94,  velocity=4.76  (continuous joint, no URDF limits)
#   gripper_moving: 1x STS3215 -> effort=2.94,  velocity=4.76

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
            solver_position_iteration_count=12, # increased from 8
            solver_velocity_iteration_count=1,  # increased from 0 for stability
        ),
        # joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        #     gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        # ),
        joint_drive=None, # replace this with the above commented out block if conversion issue isn't fixed
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        # Exact joint names extracted
        joint_pos={
            "base_yaw": 0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch": -0.0,
            "wrist_pitch": 0.0, # changed from 1.57 for testing so it doesn't conflict with reaching
            "wrist_roll": -0.0,
            "gripper_moving": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 1x STS3215
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_yaw"],
            effort_limit_sim=2.94,
            velocity_limit_sim=4.76,
            stiffness=200.0,
            damping=80.0,
        ),
        # 2x STS3250 — higher torque and speed than STS3215
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pitch"],
            effort_limit_sim=9.80,
            velocity_limit_sim=8.05,
            stiffness=400.0,
            damping=160.0,
        ),
        # 2x STS3215
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_pitch"],
            effort_limit_sim=5.88,
            velocity_limit_sim=4.76,
            stiffness=240.0,
            damping=90.0,
        ),
        # 1x STS3215 each — split stiffness/damping since wrist_roll bears less load
        # Note: wrist_roll is a continuous joint in the URDF (no limits).
        # Consider adding joint position limits in your env config to prevent
        # unbounded spinning during training.
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_pitch", "wrist_roll"],
            effort_limit_sim=2.94,
            velocity_limit_sim=4.76,
            stiffness={
                "wrist_pitch": 80.0,
                "wrist_roll":  50.0,
            },
            damping={
                "wrist_pitch": 30.0,
                "wrist_roll":  20.0,
            },
        ),
        # 1x STS3215
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_moving"],
            effort_limit_sim=2.94,
            velocity_limit_sim=4.76,
            stiffness=60.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)