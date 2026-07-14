"""
This script demonstrates how to load and simulate the Galbot One Golf articulation using Isaac Lab.

The GALBOT_ONE_GOLF_CFG is created using the github galbot foxtrot .usd.

The galbot in the simulation is imported as an articulation.

.. code-block:: bash

    # Usage
    # while the isaaclab conda env is activated
    ./scripts/run_main.sh

    # (sets PYTHONPATH=src so isaacsim_interface.* and rdt_inference.* resolve, then
    # launches this file through IsaacLab's own isaaclab.sh; see scripts/run_main.sh
    # if your IsaacLab checkout isn't at the default ~/lab/IsaacLab)

"""

"""Launch Isaac Sim Simulator first."""

import argparse


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to load the Galbot articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# this script always spawns wrist cameras, so rendering must be enabled regardless
# of whether --enable_cameras was passed on the command line
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import socket
import time
from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext

# Original is GALBOT_ONE_CHARLIE_CFG
from isaacsim_interface.robots.galbot_golf import (
    RDT_JOINT_NAME,
)

from isaacsim_interface.scenes.default import design_scene
from isaacsim_interface.interface import (
    get_joint_position,
    save_camera_image,
    set_base_velocity,
    set_default_joint,
    set_joint_position,
)
from rdt_inference.rdt_ipc import (
    CHUNK_SIZE,
    DEFAULT_CONNECT_HOST,
    DEFAULT_PORT,
    RPC_TIMEOUT_S,
    encode_image,
    parse_addr,
    recv_msg,
    send_msg,
)

# Physics runs at 200 Hz (sim_dt=0.005s); RDT was fine-tuned at a 30 Hz control frequency.
# 200 / 30 ~= 6.67, so a new control tick fires every 7 physics steps (~28.6 Hz effective).
PHYSICS_STEPS_PER_CTRL = 7


def connect_to_rdt_server(addr: tuple[str, int], timeout: float = RPC_TIMEOUT_S) -> socket.socket:
    """Connects to the RDT inference server, retrying until it's up or `timeout` elapses."""
    host, port = addr
    print(f"[INFO]: Waiting for RDT inference server at {host}:{port} ...")
    deadline = time.time() + timeout
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            sock.settimeout(RPC_TIMEOUT_S)
            print("[INFO]: Connected to RDT inference server.")
            return sock
        except (ConnectionRefusedError, OSError):
            sock.close()
            if time.time() > deadline:
                raise RuntimeError(
                    f"Could not connect to RDT inference server at {host}:{port} after "
                    f"{timeout:.0f}s. Start it first with:\n"
                    "  conda activate rdt && ./scripts/start_rdt_server.sh"
                )
            time.sleep(1.0)


def run_simulator(
    sim: SimulationContext,
    robot: Articulation,
    left_wrist_camera: Camera,
    right_wrist_camera: Camera,
    head_camera: Camera,
    sock: socket.socket,
):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    print(sim_dt)
    count = 0

    # 2-slot history of {"images": {...}, "proprio": Tensor(19,)}, seeded with an
    # all-None dummy entry so the very first control tick still has a "previous"
    # observation to pair with (policy.step() pads None images with the background image).
    obs_window = deque(maxlen=2)
    obs_window.append({
        "images": {"head": None, "left_wrist": None, "right_wrist": None},
        "proprio": None,
    })
    action_chunk = None
    chunk_idx = 0

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()

        # -----------------------------------------------------------------
        # Inference logic goes here
        if (count % PHYSICS_STEPS_PER_CTRL == 0) and (count >= 10):
            # 1. Capture the current observation (proprio + 3 camera images)
            arm_grip_pos = get_joint_position(robot, RDT_JOINT_NAME).to(torch.float32).cpu()
            base_vx_vy = robot.data.root_lin_vel_b[0, :2].to(torch.float32).cpu()
            base_w = robot.data.root_ang_vel_b[0, 2:3].to(torch.float32).cpu()
            proprio = torch.cat([arm_grip_pos, base_vx_vy, base_w])

            obs_window.append({
                "images": {
                    "head": save_camera_image(head_camera),
                    "right_wrist": save_camera_image(right_wrist_camera),
                    "left_wrist": save_camera_image(left_wrist_camera),
                },
                "proprio": proprio,
            })

            # 2. Re-infer a fresh 64-step action chunk at chunk boundaries only
            if action_chunk is None:
                prev_imgs = obs_window[-2]["images"]
                cur_imgs = obs_window[-1]["images"]
                send_msg(sock, {
                    "proprio": obs_window[-1]["proprio"].tolist(),
                    "images": {
                        "head_prev": encode_image(prev_imgs["head"]),
                        "right_wrist_prev": encode_image(prev_imgs["right_wrist"]),
                        "left_wrist_prev": encode_image(prev_imgs["left_wrist"]),
                        "head_cur": encode_image(cur_imgs["head"]),
                        "right_wrist_cur": encode_image(cur_imgs["right_wrist"]),
                        "left_wrist_cur": encode_image(cur_imgs["left_wrist"]),
                    },
                })
                print("Sent message")
                action_chunk = recv_msg(sock)["action_chunk"]
                print(f"Received message with #actions={len(action_chunk)}")
                current_time = time.time()
                chunk_idx = 0

            # 3. Execute one action from the chunk this control tick
            action = action_chunk[chunk_idx]
            # if count >= 1000: print(action)
            set_joint_position(robot, action[:16], RDT_JOINT_NAME, set_PD=True)
            # Set head to default
            set_joint_position(robot, [0.3], ["head_joint2"], set_PD=True)
            set_base_velocity(robot, action[16], action[17], action[18])
            # print(chunk_idx)
            # print(f"Time elapsed one step is {time.time() - current_time}")

            chunk_idx += 1
            if chunk_idx >= CHUNK_SIZE:
                action_chunk = None
            
            # if (time.time() - current_time) >= CHUNK_SIZE/30:
            #     action_chunk = None
        # -----------------------------------------------------------------

        count += 1
        # update buffers
        robot.update(sim_dt)
        left_wrist_camera.update(sim_dt)
        right_wrist_camera.update(sim_dt)
        head_camera.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    # default dt is 0.005
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cpu")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])
    # design scene
    robot, left_wrist_camera, right_wrist_camera, head_camera = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    print()

    # Set joints velocity to 0
    set_default_joint(robot)

    # Connect to the RDT-1B inference server (runs separately, in the `rdt` conda env --
    # see src/rdt_inference/rdt_server.py)
    addr = parse_addr(os.environ.get("RDT_SERVER_ADDR"), f"{DEFAULT_CONNECT_HOST}:{DEFAULT_PORT}")
    sock = connect_to_rdt_server(addr)

    try:
        # robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
        # robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
        initial_pos2 = [1.171875, -1.484375, -0.5546875, -1.734375, -0.10205078125, 0.1884765625, 0.08447265625, -1.421875, 1.1484375, 0.3984375, 2.046875, 0.142578125, -0.01025390625, 0.2333984375, 0.796875, 0.88671875]
        initial_pos = [1.7890625, -1.453125, -0.478515625, -2.34375, 0.060546875, 0.1328125, -0.11181640625, -1.765625, 1.234375, 0.5078125, 2.25, 0.14453125, -0.029296875, 0.302734375, 0.91015625, 0.890625]
        set_joint_position(robot, initial_pos, RDT_JOINT_NAME)
        # Run the simulator
        run_simulator(sim, robot, left_wrist_camera, right_wrist_camera, head_camera, sock)
    finally:
        sock.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
