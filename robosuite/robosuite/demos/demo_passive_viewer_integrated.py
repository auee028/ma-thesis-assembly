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

from robosuite.custom_utils.assembly_llm import AssemblyLLM
from robosuite.custom_utils.assembly_spatial_graph import AssemblySpatialGraph
from robosuite.custom_utils.assembly_planner import AssemblyPlanner
from robosuite.custom_utils.assembly_utils import read_file, match_block_names, save_cam_image
from robosuite.custom_utils.assembly_utils import ENVNAME2SHAPES, RGBA2COLORNAME, get_color_name

# Load config file
config_path = "../configs/assembly_configs.json"
configs = read_file(config_path)

# Import the controller config file as a dict
controller_config = load_composite_controller_config(controller=configs["controller"]["config_fpath"])

# Create environment instance with offscreen rendering enabled
env_config = configs["env"]

env = suite.make(
    env_name=env_config["env_name"],  # Task: "Pyramid", "PyramidSixBlocks", "TowerTwoBlocks", "TowerFiveBlocks", "House", "BigHouse", "AlphaBlock", etc.
    robots=env_config["robots"],  # Robot: "Sawyer", "Jaco", etc.
    # gripper_types="default",  # Use default grippers per robot arm
    controller_configs=controller_config,   # BASIC: arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=env_config["has_renderer"],  # No on-screen rendering
    has_offscreen_renderer=env_config["has_offscreen_renderer"],  # Enable offscreen rendering
    use_camera_obs=env_config["use_camera_obs"],  # Enable camera observations
    camera_heights=env_config["camera_heights"],
    camera_widths=env_config["camera_widths"],
    camera_names=env_config["camera_names"],
    camera_depths=env_config["camera_depths"],
    camera_segmentations=env_config["camera_segmentations"],
    hard_reset=env_config["hard_reset"],
    mujoco_passive_viewer=env_config["mujoco_passive_viewer"],
    horizon=env_config["horizon"],    # Put a larger number for the limit of horizon-based termination
)

# Adjust pose of 'sideview' camera
cam_id = env.sim.model.camera_name2id("sideview")
env.sim.model.cam_pos[cam_id] = np.array([0.0, 0.9, 1.2])

# Access the geometries' names and ids
geom_names = env.sim.model.geom_names

# Get information of interested objects
shapes = ENVNAME2SHAPES.get(env_config["env_name"])
block_geom_names = tuple(
    geom_name for geom_name in geom_names 
    if any(shape in geom_name for shape in shapes) and '_vis' in geom_name
)
block_geom_ids = tuple(
    env.sim.model.geom_name2id(geom_name) for geom_name in block_geom_names
)
block_colors = tuple(
    get_color_name(env.sim.model.geom_rgba[geom_id], RGBA2COLORNAME)
    for geom_id in block_geom_ids
)
block_info = {
    geom_name.replace("visual", "vis"): {"geom_id": geom_id, "color": color}
    for geom_name, geom_id, color in zip(block_geom_names, block_geom_ids, block_colors)
}
print("Block info: ", block_info)

# Define camera views to capture
camera_names = env.sim.model.camera_names

# Create a directory to save images
save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}')
os.mkdir(save_dir)

# Reset the environment
env.reset()

# Capture and save images from multiple views before assembling
obs = env._get_observations()
for cam_name in camera_names:
    # Ensure camera's RGB channel exists in observation and then get and save the RGB image
    rgb_cam_name = cam_name + "_image"
    if rgb_cam_name in obs.keys():
        rgb_img = obs[rgb_cam_name]  # Get image from the observation
        save_path = os.path.join(save_dir, f"before_assembling_{cam_name}_frame.png")
        save_cam_image(rgb_img, save_path)
        
        # Save for target rgb image fed together with target query
        if configs["task"]["target_cam"] == cam_name:
            os.makedirs(configs["task"]["target_rgb_dir"], exist_ok=True)
            save_path = configs["task"]["target_rgb_path"]
            save_cam_image(rgb_img, save_path)

# Set up LLM
llm = AssemblyLLM(configs)
detected_blocks, instructions = llm()    # returns detected_blocks and assembly_structure

# Map colors to available geom_names
color_to_geom_names = defaultdict(list)
for geom_name, info in block_info.items():
    color = info["color"]
    if color:
        color_to_geom_names[color].append(geom_name)

# Match detection results and real block names in robosuite env
block_matches = match_block_names(detected_blocks, color_to_geom_names)
print("Block matches: ", block_matches)

# Create spatial graph and get assembly order
print("* * *")
print("Instructions: ", instructions)
   
g = AssemblySpatialGraph()
spatial_graph, assembly_order = g(instructions)

# Plan and execute
planner = AssemblyPlanner(env, block_matches)
planner(spatial_graph, assembly_order)

# Capture and save images from multiple views after assembling
obs = env._get_observations()
for cam_name in camera_names:
    # Ensure camera's RGB channel exists in observation and then get and save the RGB image
    rgb_cam_name = cam_name + "_image"
    if rgb_cam_name in obs.keys():
        rgb_img = obs[rgb_cam_name]  # Get image from the observation
        save_path = os.path.join(save_dir, f"after_assembling_{cam_name}_frame.png")
        save_cam_image(rgb_img, save_path)
            
env.close()

