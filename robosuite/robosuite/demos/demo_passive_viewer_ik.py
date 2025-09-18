import numpy as np
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map
from robosuite.controllers import load_composite_controller_config
# from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import cv2  # For saving images
import os
from datetime import datetime
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from robosuite.custom_utils.assembly_utils import *

# # Load WholeBodyIK controller config
# controller_config = load_composite_controller_config(controller="WHOLE_BODY_IK")

# Path to config file
controller_fpath = "../custom_utils/whole_body_ik_assembly.json"

# Import the file as a dict
controller_config = load_composite_controller_config(controller=controller_fpath)

# Create environment instance with offscreen rendering enabled
env = suite.make(
    env_name="Stack",  # Task: "Pyramid", "PyramidSixBlocks", "TowerTwoBlocks", "TowerFiveBlocks", "House", "BigHouse", "AlphaBlock", etc.
    robots="UR5e",  # Robot: "Sawyer", "Jaco", etc.
    # gripper_types="default",  # Use default grippers per robot arm
    controller_configs=controller_config,   # BASIC: arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=False,  # No on-screen rendering
    has_offscreen_renderer=True,  # Enable offscreen rendering
    use_camera_obs=True,  # Enable camera observations
    camera_heights=256,
    camera_widths=256,
    # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    # camera_depths=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_names=['agentview'],
    camera_depths=['agentview'],
    camera_segmentations='element',
    hard_reset=False,
    mujoco_passive_viewer=True,
)

'''

options = {
    "env_name": "Stack", 
    "robots": "UR5e",
}

controller_name = "IK_POSE"
arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
options["controller_configs"] = refactor_composite_controller_config(
    arm_controller_config, options["robots"], ["right"]
)

# Create environment instance with offscreen rendering enabled
env = suite.make(
    # env_name="Stack",  # Task: "Pyramid", "PyramidSixBlocks", "TowerTwoBlocks", "TowerFiveBlocks", "House", "BigHouse", "AlphaBlock", etc.
    # robots="UR5e",  # Robot: "Sawyer", "Jaco", etc.
    **options,
    # gripper_types="default",  # Use default grippers per robot arm
    # controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=False,  # No on-screen rendering
    has_offscreen_renderer=True,  # Enable offscreen rendering
    use_camera_obs=True,  # Enable camera observations
    camera_heights=256,
    camera_widths=256,
    # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    # camera_depths=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_names=['agentview'],
    camera_depths=['agentview'],
    camera_segmentations='element',
    hard_reset=False,
    mujoco_passive_viewer=True,
)
'''


obs = env.reset()
print(obs)

robot = env.robots[0]
# init_qpos = robot.init_qpos
eef_pos = obs["robot0_eef_pos"]
# eef_quat = obs["robot0_eef_quat"]
eef_quat = obs["robot0_eef_quat_site"]  # "robot0_eef_quat_site" for IK
# eef_quat = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])  # wxyz to xyzw
# eef_rpy = R.from_quat(eef_quat, scalar_first=True).as_euler('xyz')  # scalar_first=True for wxyz order
from robosuite.utils.transform_utils import quat2axisangle
eef_rpy = quat2axisangle(eef_quat)
#eef_rpy = np.array([np.pi, 0.0, 0.0])  # roll, pitch, yaw
print(eef_pos, eef_quat)
print(eef_rpy)
gripper_state = [1.0]

for i in range(50):
    # action = np.concatenate([init_qpos, gripper_state])
    # eef_pos[0] += 0.01
    action = np.concatenate([eef_pos, eef_rpy, gripper_state])
    # action = np.concatenate([obs['robot0_gripper_qpos'], gripper_state])
    obs, reward, done, info = env.step(action)
            
env.close()
