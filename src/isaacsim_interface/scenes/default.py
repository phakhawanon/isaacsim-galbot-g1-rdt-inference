"""Scene construction for the Galbot One Golf simulation."""

import omni.usd
import omni.physx.scripts.utils as physx_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera

from isaacsim_interface.robots.galbot_golf import (
    GALBOT_ONE_GOLF_CFG,
    GALBOT_ONE_GOLF_LEFT_WRIST_CAMERA_CFG,
    GALBOT_ONE_GOLF_RIGHT_WRIST_CAMERA_CFG,
    GALBOT_ONE_GOLF_HEAD_CAMERA_CFG,
)

ROBOT_PRIM_PATH = "/World/Galbot"


def design_scene() -> tuple[Articulation, Camera, Camera, Camera]:
    """Designs the scene."""
    stage = omni.usd.get_context().get_stage()

    # Ground-plane
    # NOTE: GroundPlaneCfg's `color` is a tint multiplied over the default grid texture's own
    # blue grid lines, so it can't be recolored to an arbitrary flat color (it stays blue-ish
    # no matter the tint). Use a flat-shaded cuboid instead for a true light-brown ground.
    ground_cfg = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.2, 0.1)),
    )
    ground_cfg.func(
        "/World/defaultGroundPlane", ground_cfg,
        translation=(0.0, 0.0, -0.05),
    )
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # # Background Cube
    # cube_cfg = sim_utils.CuboidCfg(
    #     size=(0.1, 10, 10),  # full size in meters (x, y, z) — this is what controls "scale"
    #     collision_props=sim_utils.CollisionPropertiesCfg(),
    #     # rigid_props=sim_utils.RigidBodyPropertiesCfg(),      # optional: only if it should be dynamic
    #     # mass_props=sim_utils.MassPropertiesCfg(mass=0.1),    # optional: only if rigid
    #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.6)),
    # )
    # cube_cfg.func(
    #     "/World/Cube", cube_cfg,
    #     translation=(1.8, 0.0, 0.0)
    # )

    # Table
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=(
            "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/"
            "Assets/simready_content/common_assets/props/danny/danny.usd"
        ),
        scale=(1.0, 1.0, 0.3),
    )
    table_cfg.func(
        "/World/Table", table_cfg,
        translation=(0.16, 0.0, 0.0),
    )
    # NOTE: UsdFileCfg's collision_props/rigid_props/mass_props only *modify* PhysX schemas that
    # already exist on the referenced asset — they can't add schemas to assets that don't ship with
    # any (like these simready render props). To reproduce the GUI's
    # "Add > Physics > Rigid Body with Colliders Preset", apply the same underlying PhysX utility
    # it uses: it applies RigidBodyAPI at the given prim and recurses over its child meshes to add
    # CollisionAPI with a convex-hull approximation.
    physx_utils.setRigidBody(
        stage.GetPrimAtPath("/World/Table"), "convexHull", kinematic=True
    )

    # Bottle
    bottle_cfg = sim_utils.UsdFileCfg(
        usd_path=(
            "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/"
            "Assets/simready_content/common_assets/props/utilityjug_a01/utilityjug_a01.usd"
        ),
        scale=(0.7, 0.7, 0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.2)),
    )
    bottle_cfg.func(
        "/World/Bottle", bottle_cfg,
        translation=(0.01, 0.2, 0.3),
    )
    physx_utils.setRigidBody(
        stage.GetPrimAtPath("/World/Bottle"), "convexHull", kinematic=False
    )

    # Cup
    cup_cfg = sim_utils.UsdFileCfg(
        usd_path=(
            "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/"
            "Assets/simready_content/common_assets/props/naturalbostonroundbottle_a01/naturalbostonroundbottle_a01.usd"
        ),
        scale=(1.0, 1.0, 1.5),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cup_cfg.func(
        "/World/Cup", cup_cfg,
        translation=(0.01, -0.2, 0.3),
    )
    physx_utils.setRigidBody(
        stage.GetPrimAtPath("/World/Cup"), "convexHull", kinematic=False
    )

    # bottle: https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/naturalbostonroundbottle_a01/naturalbostonroundbottle_a01.usd

    # Robot
    robot = Articulation(GALBOT_ONE_GOLF_CFG.replace(prim_path=ROBOT_PRIM_PATH))

    # Disable ActionGraph
    action_graph_prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/ActionGraph")
    if action_graph_prim.IsValid():
        action_graph_prim.SetActive(False)

    # Wrist cameras
    left_wrist_camera = Camera(GALBOT_ONE_GOLF_LEFT_WRIST_CAMERA_CFG)
    right_wrist_camera = Camera(GALBOT_ONE_GOLF_RIGHT_WRIST_CAMERA_CFG)
    head_camera = Camera(GALBOT_ONE_GOLF_HEAD_CAMERA_CFG)

    return robot, left_wrist_camera, right_wrist_camera, head_camera
