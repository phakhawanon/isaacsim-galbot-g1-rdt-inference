# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Galbot One Golf humanoid robot.

The following configuration parameters are available:

* :obj:`GALBOT_ONE_GOLF_CFG`: The galbot_one_golf humanoid robot, with a parallel gripper on both arms.
* :obj:`GALBOT_ONE_GOLF_LEFT_WRIST_CAMERA_CFG`: Camera mounted on the left wrist.
* :obj:`GALBOT_ONE_GOLF_RIGHT_WRIST_CAMERA_CFG`: Camera mounted on the right wrist.

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg

##
# Configuration
##

# Path to the galbot_one_golf_description checkout (USD + meshes), published at
# https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description -- override with the
# GALBOT_ASSETS_ROOT env var if yours isn't at the default below (scripts/run_main.sh checks
# and exports this for you).
GALBOT_ASSETS_ROOT = os.environ.get("GALBOT_ASSETS_ROOT", os.path.expanduser("~/galbot_one_golf_description"))

GALBOT_PRIM_PATH = "/World/Galbot"

JOINT_NAME_TO_ID_DICT = {
    'head_joint1': 1,
    'left_arm_joint1': 2,
    'right_arm_joint1': 3,
    'leg_joint4': 4,
    'head_joint2': 5,
    'left_arm_joint2': 6,
    'right_arm_joint2': 7,
    'leg_joint3': 8,
    'left_arm_joint3': 9,
    'right_arm_joint3': 10,
    'leg_joint2': 11,
    'left_arm_joint4': 12,
    'right_arm_joint4': 13,
    'leg_joint1': 14,
    'left_arm_joint5': 15,
    'right_arm_joint5': 16,
    'wheel1_joint': 17,
    'wheel2_joint': 18,
    'wheel3_joint': 19,
    'wheel4_joint': 20,
    'left_arm_joint6': 21,
    'right_arm_joint6': 22,
    'left_arm_joint7': 63,
    'right_arm_joint7': 64,
    'left_gripper_joint': 65,
    'right_gripper_joint': 69,
}

ATLP_JOINT_NAME = [
    'left_arm_joint1',
    'left_arm_joint2',
    'left_arm_joint3',
    'left_arm_joint4',
    'left_arm_joint5',
    'left_arm_joint6',
    'left_arm_joint7',
    'right_arm_joint1',
    'right_arm_joint2',
    'right_arm_joint3',
    'right_arm_joint4',
    'right_arm_joint5',
    'right_arm_joint6',
    'right_arm_joint7',
    'leg_joint1',
    'leg_joint2',
    'leg_joint3',
    'leg_joint4',
    'head_joint1',
    'head_joint2',
    'left_gripper_joint',
    'right_gripper_joint',
]

RDT_JOINT_NAME = [
    'left_arm_joint1',
    'left_arm_joint2',
    'left_arm_joint3',
    'left_arm_joint4',
    'left_arm_joint5',
    'left_arm_joint6',
    'left_arm_joint7',
    'right_arm_joint1',
    'right_arm_joint2',
    'right_arm_joint3',
    'right_arm_joint4',
    'right_arm_joint5',
    'right_arm_joint6',
    'right_arm_joint7',
    'left_gripper_joint',
    'right_gripper_joint',
]


GALBOT_ONE_GOLF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(GALBOT_ASSETS_ROOT, "usd", "galbot_one_golf.usda"),
        variants={"Physics": "PhysX"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Default Joint Pose
            "leg_joint1": 0.25,
            "leg_joint2": 1.2,
            "leg_joint3": 0.8,
            "leg_joint4": 0.0,
            "leg_joint5": 0.0,  # TODO: no charlie equivalent, needs tuning
            "head_joint1": 0.0,
            "head_joint2": 0.0,
            "left_arm_joint1": -0.5480,
            "left_arm_joint2": -0.6551,
            "left_arm_joint3": 2.407,
            "left_arm_joint4": 1.3641,
            "left_arm_joint5": -0.4416,
            "left_arm_joint6": 0.1168,
            "left_arm_joint7": 1.2308,
            "left_gripper_joint": 0.0,
            "right_arm_joint1": 0.1535,
            "right_arm_joint2": 1.0087,
            "right_arm_joint3": 0.0895,
            "right_arm_joint4": 1.5743,
            "right_arm_joint5": -0.2422,
            "right_arm_joint6": -0.0009,
            "right_arm_joint7": -0.9143,
            "right_gripper_joint": 0.0,

            # New Joint Pose
            # "leg_joint1": 0.25,
            # "leg_joint2": 1.2,
            # "leg_joint3": 0.8,
            # "leg_joint4": 0.0,
            # "leg_joint5": 0.0,
            # "head_joint1": 0.0,
            # "head_joint2": 0.0,
            # "left_arm_joint1": 2,
            # "left_arm_joint2": -1.2,
            # "left_arm_joint3": 0.5,
            # "left_arm_joint4": -2.2,
            # "left_arm_joint5": 0.2,
            # "left_arm_joint6": 0.0,
            # "left_arm_joint7": 0.0,
            # "left_gripper_joint": 0.0,
            # "right_arm_joint1": 0.1535,
            # "right_arm_joint2": 1.0087,
            # "right_arm_joint3": 0.0895,
            # "right_arm_joint4": 1.5743,
            # "right_arm_joint5": -0.2422,
            # "right_arm_joint6": -0.0009,
            # "right_arm_joint7": -0.9143,
            # "right_gripper_joint": 0.0,
        },
        pos=(-0.6, 0.0, 0.1),
    ),
    # PD parameters are read from USD file with Gain Tuner
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "leg": ImplicitActuatorCfg(
            joint_names_expr=["leg_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_joint"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=None,
            damping=None,
        ),
        # covers the 3 driven wheel joints and ~50 passive omni-wheel roller joints;
        # zero gains let the base roll freely instead of holding the USD's baked-in
        # 100000-damping brake on the driven wheels
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel.*"],
            velocity_limit_sim=None,
            effort_limit_sim=None,
            stiffness=0,
            damping=10000.0,
        ),
    },
)
"""Configuration of Galbot_one_golf humanoid using implicit actuator models, with a gripper on both arms."""


##
# Wrist cameras
##
# NOTE: "right_arm_wrist_camera_stand" and "right_wrist_camera_link" (and their left-arm
# counterparts) do not exist in the source USD. Isaac Sim will auto-create them as
# identity-transform placeholder Xforms when the camera is spawned, so the `offset` below
# is currently the only thing positioning/orienting the camera relative to
# "right_arm_link7"/"left_arm_link7" and will need tuning once real mount geometry exists.

_WRIST_CAMERA_SPAWN_CFG = sim_utils.PinholeCameraCfg(
    # focal_length=24.0,
    focal_length=2.213,
    focus_distance=400.0,
    # horizontal_aperture=20.955,
    horizontal_aperture=4.2,
    clipping_range=(0.005, 3.0),
)
_WRIST_CAMERA_OFFSET = CameraCfg.OffsetCfg(
    pos=(0.0, 0.0, -0.003416),
    rot=(0,-0.707,-0.707,0),
    convention="opengl",
)

GALBOT_ONE_GOLF_LEFT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path=f"{GALBOT_PRIM_PATH}/left_arm_link7/left_arm_wrist_camera_stand/left_wrist_camera_link/left_wrist_camera",
    update_period=0.0,
    height=368,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=_WRIST_CAMERA_SPAWN_CFG,
    offset=_WRIST_CAMERA_OFFSET,
)
"""Camera mounted under left_arm_link7/left_arm_wrist_camera_stand/left_wrist_camera_link."""

GALBOT_ONE_GOLF_RIGHT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path=f"{GALBOT_PRIM_PATH}/right_arm_link7/right_arm_wrist_camera_stand/right_wrist_camera_link/right_wrist_camera",
    update_period=0.0,
    height=368,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=_WRIST_CAMERA_SPAWN_CFG,
    offset=_WRIST_CAMERA_OFFSET,
)

"""Camera mounted under right_arm_link7/right_arm_wrist_camera_stand/right_wrist_camera_link."""

_HEAD_CAMERA_SPAWN_CFG = sim_utils.PinholeCameraCfg(
    # focal_length=24.0,
    focal_length=2,
    focus_distance=400.0,
    # horizontal_aperture=20.955,
    horizontal_aperture=6.2,
    clipping_range=(0.005, 3.0),
)
_HEAD_CAMERA_OFFSET = CameraCfg.OffsetCfg(
    pos=(0.085,0.0,0.0),
    rot=(0.0,-0.707,0.0,0.707),
    convention="opengl",
)


GALBOT_ONE_GOLF_HEAD_CAMERA_CFG = CameraCfg(
    prim_path=f"{GALBOT_PRIM_PATH}/head_link2/head_end_effector_mount_link/head_camera",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=_HEAD_CAMERA_SPAWN_CFG,
    offset=_HEAD_CAMERA_OFFSET,
)
