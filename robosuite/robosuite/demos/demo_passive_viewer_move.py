import numpy as np
import cv2
import os
from datetime import datetime
import open3d as o3d
from collections import defaultdict

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map
from robosuite.utils.transform_utils import quat2axisangle

from robosuite.custom_utils.assembly_spatial_graph import AssemblySpatialGraph
from robosuite.custom_utils.assembly_planner import AssemblyPlanner
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors
from robosuite.custom_utils.assembly_utils import COLOR_RGBA2NAME, get_color_name


# Path to config file
controller_fpath = "../custom_utils/whole_body_ik_assembly.json"

# Import the file as a dict
controller_config = load_composite_controller_config(controller=controller_fpath)

# Create environment instance with offscreen rendering enabled
env = suite.make(
    env_name="PyramidSixBlocks",  # Task: "Pyramid", "PyramidSixBlocks", "TowerTwoBlocks", "TowerFiveBlocks", "House", "BigHouse", "AlphaBlock", etc.
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
    camera_names=['frontview', 'birdview', 'agentview', 'sideview'],
    camera_depths=['frontview', 'birdview', 'agentview', 'sideview'],
    camera_segmentations='element',
    hard_reset=False,
    mujoco_passive_viewer=True,
    horizon=5000,    # Put a larger number for the limit of horizon-based termination
)

# Adjust pose of 'sideview' camera
cam_id = env.sim.model.camera_name2id("sideview")
env.sim.model.cam_pos[cam_id] = np.array([0.0, 0.9, 1.2])

# Access the geometries' names and ids
geom_names = env.sim.model.geom_names

# Get information of interested objects
shapes = ['cube', 'triangle']
block_geom_names = tuple(
    geom_name for geom_name in geom_names 
    if any(shape in geom_name for shape in shapes) and '_vis' in geom_name
)
block_geom_ids = tuple(
    env.sim.model.geom_name2id(geom_name) for geom_name in block_geom_names
)
block_colors = tuple(
    get_color_name(env.sim.model.geom_rgba[geom_id], COLOR_RGBA2NAME)
    for geom_id in block_geom_ids
)
block_info = {
    geom_name: {"geom_id": geom_id, "color": color}
    for geom_name, geom_id, color in zip(block_geom_names, block_geom_ids, block_colors)
}
print("Block info: ", block_info)

# Define camera views to capture
camera_names = env.sim.model.camera_names
# print(camera_names)  # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')

# Create a directory to save images
save_dir = f'/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(save_dir, exist_ok=True)

# Reset the environment
env.reset()
# obs = env.reset()
# print(obs)

# default_eef_pos = obs["robot0_eef_pos"]
# default_eef_quat = obs["robot0_eef_quat_site"]  # xyzw order, "robot0_eef_quat_site" for IK
# default_eef_rpy = quat2axisangle(default_eef_quat)  # roll, pitch, yaw
# default_gripper_state = [-1.0]  # 1.0: close, -1.0: open

detected_blocks = [
    {"id": "block0", "color": "green"},
    {"id": "block1", "color": "orange"},
    {"id": "block2", "color": "purple"},
    {"id": "block3", "color": "yellow"},
    {"id": "block4", "color": "red"},
    {"id": "block5", "color": "blue"}
]
instructions = [
    "block5 is on the table.",
    "block0 is left of block5.",
    "block4 is right of block5.",
    "block1 is above block0 and block5.",
    "block2 is above block5 and block4.",
    "block3 is above block1 and block2."
]

# Map colors to available geom_names
color_to_geom_names = defaultdict(list)
for geom_name, info in block_info.items():
    color = info["color"]
    if color:
        color_to_geom_names[color].append(geom_name)
# print(color_to_geom_names)

# Match detected blocks with environment observations
block_matches = {}
for i, det in enumerate(detected_blocks):
    block_id = det["id"]
    block_color = det["color"]
    
    if block_color in color_to_geom_names and color_to_geom_names[block_color]:
        # Pop the first available geom_name (removes it from the list)
        geom_name = color_to_geom_names[block_color].pop(0)
        block_matches[block_id] = geom_name.replace('_g0_vis', '')
    else:
        print(f"Warning: No unassigned geom_name left for {block_id} (color: {block_color})")   
# print(block_matches)

# Create spatial graph and get assembly order
print("Instructions: ", instructions)
   
g = AssemblySpatialGraph()
spatial_graph, assembly_order = g(instructions)

# Plan and execute
planner = AssemblyPlanner(env, block_matches)
planner(spatial_graph, assembly_order)

# Capture and save images from multiple views
obs = env._get_observations()
for cam_name in camera_names:
    # Ensure camera's RGB channel exists in observation and then get and save the RGB image
    rgb_cam_name = cam_name + "_image"
    if rgb_cam_name in obs.keys():
        rgb_img = obs[rgb_cam_name]  # Get image from the observation
        rgb_img = np.flipud(rgb_img)  # Flip vertically to match normal image orientation
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        save_path = os.path.join(save_dir, f"{cam_name}_frame_rgb.png")
        cv2.imwrite(save_path, rgb_img)  # Save the image
        print(f"Saved: {save_path}")
            
env.close()

