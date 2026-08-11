import os
import time
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map
from robosuite.utils.transform_utils import quat2axisangle, mat2quat, axisangle2quat

from robosuite.custom_utils.assembly_utils import *


# Load config file
config_path = "../configs/fmb_configs.json"
configs = read_file(config_path)

# Import the controller config file as a dict
controller_config = load_composite_controller_config(controller=configs["controller"]["config_fpath"])

# Create environment instance with offscreen rendering enabled
env_config = configs["env"]

env = suite.make(
    env_name=env_config["env_name"],
    robots=env_config["robots"],
    controller_configs=controller_config,
    has_renderer=env_config["has_renderer"],
    has_offscreen_renderer=env_config["has_offscreen_renderer"],
    use_camera_obs=env_config["use_camera_obs"],
    camera_heights=env_config["camera_heights"],
    camera_widths=env_config["camera_widths"],
    camera_names=env_config["camera_names"],
    camera_depths=env_config["camera_depths"],
    camera_segmentations=env_config["camera_segmentations"],
    hard_reset=env_config["hard_reset"],
    mujoco_passive_viewer=env_config["mujoco_passive_viewer"],
    horizon=env_config["horizon"],
    render_collision_mesh=False,
)

# Define camera views to capture
camera_names = env.sim.model.camera_names
geom_names = env.sim.model.geom_names

# Reset the environment (the viewer is created at this point)
obs = env.reset()

# Disable collision geometry (group 0) rendering in the Passive Viewer
if hasattr(env, "viewer") and env.viewer is not None:
    if hasattr(env.viewer, "vopt"):
        env.viewer.vopt.geomgroup[0] = 0

# -------------------------------------------------------------
# 1. Set viewer options after reset() (disable collision visualization for group 0)
# -------------------------------------------------------------
def setup_viewer_options(env):
    viewer = getattr(env, "viewer", None)
    if viewer is None and hasattr(env, "mujoco_passive_viewer"):
        viewer = env.mujoco_passive_viewer

    if viewer is not None:
        # Access vopt/opt depending on the Passive Viewer structure
        vopt = getattr(viewer, "vopt", getattr(viewer, "opt", None))
        if vopt is not None:
            vopt.geomgroup[0] = 0  # Turn off collision geometry (group 0) gray lines
            vopt.geomgroup[1] = 1  # Turn on visual geometry (group 1)
            vopt.geomgroup[3] = 0  # Turn off group 3 if any geometry remains

setup_viewer_options(env)

# Adjust the camera pose so that the 'agentview' can capture all blocks on the table
target_cam = "birdview"
cam_id = env.sim.model.camera_name2id(target_cam)
env.sim.model.cam_pos[cam_id] = np.array([-0.2, -0.15, 1.13])

axisangle = np.array([np.pi / 2, 0.0, 0.0])
env.sim.model.cam_quat[cam_id] = axisangle2quat(axisangle)

for i in range(env.sim.model.nsite):
    site_name = env.sim.model.site_id2name(i)
    site_id = env.sim.model.site_name2id(site_name)
    site_pos = env.sim.data.site_xpos[site_id]
    print(site_name, site_pos)

try:
    while True:
        # Continuously ensure that the group settings are applied
        setup_viewer_options(env)

        site_name = "gripper0_right_grip_site"
        site_id = env.sim.model.site_name2id(site_name)

        pos = env.sim.data.site_xpos[site_id].copy()
        mat = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()

        quat = mat2quat(mat)
        axis_angle = quat2axisangle(quat)

        action = np.concatenate([
            pos,
            axis_angle,
            [0.0],
        ])

        obs, reward, done, info = env.step(action)

        # Synchronize the Passive Viewer display
        if hasattr(env, "viewer") and env.viewer is not None:
            if hasattr(env.viewer, "sync"):
                env.viewer.sync()

except KeyboardInterrupt as e:
    print("\nStopping simulation...")

# Save the image of the target camera view
rgb_cam_name = target_cam + "_image"
env.sim.forward()
obs = env._get_observations()
rgb_img = obs[rgb_cam_name]
save_path = "capture.png"
save_cam_image(rgb_img, save_path)
print(f"Image saved: {save_path}")

env.close()