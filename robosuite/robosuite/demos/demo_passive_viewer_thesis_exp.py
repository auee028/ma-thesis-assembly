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
from robosuite.custom_utils.assembly_utils import *

# Load config file
config_path = "../configs/thesis_exp_configs.json"
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
        
# Define camera views to capture
camera_names = env.sim.model.camera_names

# Access the geometries' names and ids
geom_names = env.sim.model.geom_names

# Create a directory to save images
os.makedirs(configs["task"]["target_rgb_dir"], exist_ok=True)

# save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}')
save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}_thesis_{configs["exp_id"]}_{env_config["env_name"]}')
if not configs["llm"]["model_config"]["top_p"]:    # if setting of top_p is null
    save_dir += "_topp1"    # top_p=1.0 as default
os.mkdir(save_dir)

# Create a directory to save input rgb and visualization
os.mkdir(os.path.join(save_dir, "assembly_vis"))
os.mkdir(os.path.join(save_dir, "assembly_input_img"))

# Save the current files
os.system(f"cp {config_path} {os.path.join(save_dir, os.path.basename(config_path))}")

system_prompt = configs["llm"]["prompt_files"]["system_prompt"]
os.system(f"cp {system_prompt} {os.path.join(save_dir, os.path.basename(system_prompt))}")

examples_prompt = configs["llm"]["prompt_files"]["examples_prompt"]
os.system(f"cp {examples_prompt} {os.path.join(save_dir, os.path.basename(examples_prompt))}")

# Reset the environment
env.reset()

"""
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
            save_path = configs["task"]["target_rgb_path"]
            save_cam_image(rgb_img, save_path)
"""

# Set up modules of LLM and Graph
llm = AssemblyLLM(configs)
g = AssemblySpatialGraph()

# Iterate
for i in range(configs["num_iter"]):
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

    # Map colors to available geom_names
    color_to_geom_names = defaultdict(list)
    for geom_name, info in block_info.items():
        color = info["color"]
        if color:
            color_to_geom_names[color].append(geom_name)
        
    # Save image of the environment in each episode
    obs = env._get_observations()
    rgb_cam_name = configs["task"]["target_cam"] + "_image"
    rgb_img = obs[rgb_cam_name]  # Get image from the observation
    save_path = configs["task"]["target_rgb_path"]
    save_cam_image(rgb_img, save_path)
    save_path = os.path.join(save_dir, "assembly_input_img", f"assembly_rgb_{i}.png")
    save_cam_image(rgb_img, save_path)
    
    # Get llm response
    # detected_blocks, instructions = llm(log_res_path=os.path.join(save_dir, "log_llm_res.txt"))    # returns detected_blocks and assembly_structure
    content = llm(log_res_path=os.path.join(save_dir, "log_llm_res.txt"))    # returns 'content' of response from LLM
    detected_blocks = content["detected_blocks"]
    
    # Match detection results and real block names in robosuite env
    block_matches = match_block_names(detected_blocks, color_to_geom_names)
    print("Block matches: ", block_matches)
        
    if "structure_relations" in content.keys():
        assembly_structure = content["structure_relations"]
        assembly_order, new_block_positions = compute_positions_from_distances(assembly_structure)
        print("* * *")
        # print("Assembly order: ", assembly_order)
        print("New block positions: ", new_block_positions)
        
    else:    # "assembly_structure" in content.keys()
        instructions = content["assembly_structure"]
        print("* * *")
        print("Instructions: ", instructions)

        # Get assembly order
        spatial_graph, assembly_order = g(instructions)
        print("Assembly order: ", assembly_order)

        # Calculate 3d coordinates of individual blocks and log them
        planner = AssemblyPlanner(env, block_matches)
        new_block_positions = planner._compute_block_positions(spatial_graph, assembly_order)
        print("New block positions: ", new_block_positions)
    
    # Get color information
    block_colors = {}
    for det in detected_blocks:
        if "block_id" in det.keys():
            det_id = det["block_id"]
        else:
            det_id = det["id"]
        det_color = det["color"]
        block_colors[det_id] = det_color
        
    # Save visualized assembly result
    save_file_path = os.path.join(save_dir, "assembly_vis", f"assembly_res_{i}.png")
    visualize_structure(
        assembly_order,
        new_block_positions,
        block_matches,
        block_colors,
        configs["task"]["target_query_structure"],
        save_file_path)
    print(save_file_path)
    print("* * *")
    
    # Reset the environment
    env.reset()

"""
# Plan and execute
planner = AssemblyPlanner(env, block_matches)
planner(spatial_graph, assembly_order)

# Adjust pose of 'sideview' camera
cam_id = env.sim.model.camera_name2id("sideview")
env.sim.model.cam_pos[cam_id] = np.array([0.0, 0.9, 1.2])

# Capture and save images from multiple views after assembling
obs = env._get_observations()
for cam_name in camera_names:
    # Ensure camera's RGB channel exists in observation and then get and save the RGB image
    rgb_cam_name = cam_name + "_image"
    if rgb_cam_name in obs.keys():
        rgb_img = obs[rgb_cam_name]  # Get image from the observation
        save_path = os.path.join(save_dir, f"after_assembling_{cam_name}_frame.png")
        save_cam_image(rgb_img, save_path)
"""

env.close()

