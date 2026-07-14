"""Joint/base control and camera helpers for the Galbot One Golf articulation."""

import numpy as np
import torch
from PIL import Image

from isaaclab.assets import Articulation
from isaaclab.sensors import Camera

from isaaclab_assets import JOINT_NAME_TO_ID_DICT


def _convert_joint_name_to_id(
    joint_name: str | list[str]
) -> str | list[str]:
    """
        Convert joint name(s) to id(s)
    """
    if isinstance(joint_name, str):
        if joint_name not in JOINT_NAME_TO_ID_DICT:
            raise KeyError("There is no such joint name")
        return JOINT_NAME_TO_ID_DICT[joint_name]

    if isinstance(joint_name, list):
        return [_convert_joint_name_to_id(name) for name in joint_name]

def save_camera_image(camera: Camera, env_index: int = 0) -> Image.Image:
    """Returns the camera's current RGB frame as a PIL Image."""
    rgb = camera.data.output["rgb"][env_index, ..., :3]
    image = rgb.cpu().numpy().astype("uint8")
    return Image.fromarray(image, mode="RGB")

def set_joint_position(
    robot: Articulation,
    joint_pos: list[float],
    joint_name: list[str],
    set_PD: bool = False,
):
    joint_id = _convert_joint_name_to_id(joint_name)
    target = torch.tensor(joint_pos, dtype=torch.float32, device=robot.device)
    if set_PD:
        robot.set_joint_position_target(target, joint_ids=joint_id, env_ids=None)
        robot.write_data_to_sim()   # must call this to actually push the buffered target
    else:
        velocity = torch.zeros_like(target)
        robot.write_joint_state_to_sim(target, velocity, joint_ids=joint_id, env_ids=None)

def set_joint_velocity(
    robot: Articulation,
    joint_vel: list[float],
    joint_name: list[str],
    set_PD: bool = False,
):
    joint_id = _convert_joint_name_to_id(joint_name)
    target = torch.tensor(joint_vel, dtype=torch.float32, device=robot.device)
    if set_PD:
        robot.set_joint_velocity_target(target, joint_ids=joint_id, env_ids=None)
        robot.write_data_to_sim()   # must call this to actually push the buffered target
    else:
        position = torch.zeros_like(target)
        robot.write_joint_state_to_sim(position, target, joint_ids=joint_id, env_ids=None)

def set_base_velocity(
    robot: Articulation,
    vx: float,
    vy: float,
    w: float,
    use_actual_wheel: bool = False,
    set_PD: bool = False,
):
    """
        ..todo::
            - use_actual_wheel=True still uses global velocity frame when commanding
    """
    if use_actual_wheel:
        set_joint_velocity(
            robot,
            base_inverse_kinematics(vx, vy, w),
            [
                "wheel1_joint",
                "wheel2_joint",
                "wheel3_joint",
                "wheel4_joint",
            ],
            set_PD=set_PD,
        )
    else:
        target = torch.tensor([vx, vy, 0, 0, 0, w], dtype=torch.float32, device=robot.device)
        robot.write_root_velocity_to_sim(target, env_ids=None)


def set_default_joint(
    robot: Articulation
):
    default_joint_vel = robot.data.default_joint_vel
    robot.set_joint_velocity_target(default_joint_vel, joint_ids=None, env_ids=None)
    robot.write_data_to_sim()   # must call this to actually push the buffered target

def get_joint_position(
    robot: Articulation,
    joint_name: list[str],
):
    positions = robot.data.joint_pos   # shape (num_envs, num_joints)
    return positions[0, _convert_joint_name_to_id(joint_name)]

def base_inverse_kinematics(
    vx: float,
    vy: float,
    w: float,
) -> tuple[float]:
    A = np.array([
        [7.07107, -7.07107, -2.496],
        [7.07107, 7.07107, -2.496],
        [-7.07107, 7.07107, -2.496],
        [-7.07107, -7.07107, -2.496],
    ])
    v = np.array([[vx, vy, w]]).T
    return -np.dot(A,v).T
