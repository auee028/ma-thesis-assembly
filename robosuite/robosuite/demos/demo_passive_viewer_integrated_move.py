import numpy as np
import os
from datetime import datetime
from collections import defaultdict

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.utils.transform_utils import quat2axisangle

from robosuite.custom_utils.structure_planner import StructurePlanner, AssemblyLLM
from robosuite.custom_utils.action_planner import ActionPlanner
from robosuite.custom_utils.action_executor import ActionExecutor

from robosuite.custom_utils.assembly_utils import *

# Load config file
config_path = "../configs/assembly_configs_integrated_move.json"
configs = read_file(config_path)

# Import the controller config file as a dict
controller_config = load_composite_controller_config(controller=configs["controller"]["config_fpath"])

# Create environment instance with offscreen rendering enabled
env_config = configs["env"]

env = suite.make(
    env_name=env_config["env_name"],  # Task: "Pyramid", "PyramidSixBlocks", "TowerTwoBlocks", "TowerFiveBlocks", "House", "BigHouse", etc.
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

# Adjust pose of 'sideview' camera to zoom into the goal area
cam_id = env.sim.model.camera_name2id("sideview")
env.sim.model.cam_pos[cam_id] = np.array([0.0, 0.9, 1.2])

# Access the geometries' names and ids
geom_names = env.sim.model.geom_names

# Make a directory to save files
save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}_{configs["exp_id"]}_{env_config["env_name"]}')
os.makedirs(save_dir, exist_ok=True)

configs["save_dir"] = save_dir

# Create a directory to save input rgb and visualization
os.mkdir(os.path.join(save_dir, "assembly_vis"))

target_dir_tmp = os.path.join(save_dir, configs["task"]["target_dir_tmp"])
os.mkdir(target_dir_tmp)

target_dir = os.path.join(save_dir, configs["task"]["target_dir"])    # For logging over episodes
os.mkdir(target_dir)

# Save the current files
os.system(f"cp ../custom_utils/assembly_llm.py {os.path.join(save_dir, 'assembly_llm.py')}")
os.system(f"cp {config_path} {os.path.join(save_dir, os.path.basename(config_path))}")

use_prim = configs["relation"]["use_primitives"]
if use_prim is True:    # When applying spatial relation primitives
    system_prompt = configs["relation"]["prim_prompt_files"]["system_prompt"]
    os.system(f"cp {system_prompt} {os.path.join(save_dir, os.path.basename(system_prompt))}")

    examples_prompt = configs["relation"]["prim_prompt_files"]["examples_prompt"]
    os.system(f"cp {examples_prompt} {os.path.join(save_dir, os.path.basename(examples_prompt))}")
else:    # When applying 3D distance relationships
    dist_examples_prompt = configs["relation"]["dist_prompt_files"]["examples_prompt"]
    os.system(f"cp {dist_examples_prompt} {os.path.join(save_dir, os.path.basename(dist_examples_prompt))}")

which_img = configs["which_img"]
example_rgb_paths = configs["relation"]["example_images"]["example_rgb_paths"]
example_depth_paths = configs["relation"]["example_images"]["example_depth_paths"]
example_mask_paths = configs["relation"]["example_images"]["example_mask_paths"]
example_rgbd_paths = configs["relation"]["example_images"]["example_rgbd_paths"]    
example_rgbseg_paths = configs["relation"]["example_images"]["example_rgbseg_paths"]

# Reset the environment
env.reset()

# Setup the high-level structure planner
if not configs["llm"]["use_llamaindex"]:
    structure_planner = StructurePlanner(configs, use_primitives=use_prim)
else:
    structure_planner = AssemblyLLM(configs)

# Iterate
for i in range(configs["num_iter"]):
    # Get information of interested objects and map colors to available geom_names
    shapes = ENVNAME2SHAPES.get(env_config["env_name"])
    block_geom_names = tuple(
        geom_name for geom_name in geom_names 
        if any(shape in geom_name for shape in shapes) and '_vis' in geom_name
    )
    block_geom_ids = tuple(
        env.sim.model.geom_name2id(geom_name) for geom_name in block_geom_names
    )
    block_rgba2colors = tuple(
        get_color_name(env.sim.model.geom_rgba[geom_id], RGBA2COLORNAME)
        for geom_id in block_geom_ids
    )
    block_info = {
        geom_name.replace("visual", "vis"): {"geom_id": geom_id, "color": color}
        for geom_name, geom_id, color in zip(block_geom_names, block_geom_ids, block_rgba2colors)
    }
    print("Block info: ", block_info)

    color_to_geom_names = defaultdict(list)
    for geom_name, info in block_info.items():
        color = info["color"]
        if color:
            color_to_geom_names[color].append(geom_name)
        
    # Save target images of the environment in each episode
    obs = env._get_observations()
    cam_name = configs["task"]["target_cam"]
    
    # Save RGB image (as default)
    rgb_cam_name = cam_name + "_image"
    rgb_img = obs[rgb_cam_name]  # Get image from the observation
    save_path = os.path.join(target_dir_tmp, configs["task"]["target_rgb_file"])
    save_cam_image(rgb_img, save_path)
    save_path = os.path.join(target_dir, f"assembly_rgb_{i}.png")
    save_cam_image(rgb_img, save_path)
    
    # Save depth image
    if which_img in ["rgb_d", "rgb_d_m"]:
        depth_cam_name = cam_name + "_depth"
        depth_img = obs[depth_cam_name]
        depth_img = get_real_depth_map(env.sim, depth_img)  # Convert MuJoCo's depth to real-world metric values
        save_path = os.path.join(target_dir_tmp, configs["task"]["target_depth_file"])
        save_cam_depth(depth_img, save_path)
        save_path = os.path.join(target_dir, f"assembly_depth_{i}.png")
        save_cam_depth(depth_img, save_path)
    
    # Save segmentation mask
    if which_img in ["rgb_m", "rgb_d_m"]:
        seg_cam_name = cam_name + "_segmentation_element"
        seg_img = obs[seg_cam_name]
        save_path = os.path.join(target_dir_tmp, configs["task"]["target_mask_file"])
        save_cam_mask(seg_img, block_geom_ids, save_path)
        save_path = os.path.join(save_dir, "assembly_input_img", f"assembly_mask_{i}.png")
        save_cam_mask(seg_img, block_geom_ids, save_path)
        
    # Save 4-channel RGB-D image
    if which_img == "fused_rgbd":
        depth_cam_name = cam_name + "_depth"
        depth_img = obs[depth_cam_name]
        depth_img = get_real_depth_map(env.sim, depth_img)  # Convert MuJoCo's depth to real-world metric values
        save_path = os.path.join(target_dir_tmp, configs["task"]["target_rgbd_file"])
        save_fused_rgbd(rgb_img, depth_img, save_path)
        save_path = os.path.join(target_dir, f"assembly_rgbd_{i}.png")
        save_fused_rgbd(rgb_img, depth_img, save_path)
    
    # Save blocks' RGB segmentation image
    if which_img == "fused_rgbm":
        seg_cam_name = cam_name + "_segmentation_element"
        seg_img = obs[seg_cam_name]
        save_path = os.path.join(target_dir_tmp, configs["task"]["target_rgbseg_file"])
        save_fused_rgbseg(rgb_img, seg_img, block_geom_ids, save_path)
        save_path = os.path.join(save_dir, "assembly_input_img", f"assembly_rgbseg_{i}.png")
        save_fused_rgbseg(rgb_img, seg_img, block_geom_ids, save_path)
    
    # Get the LLM response for the blueprint
    blueprint = structure_planner(
        log_res_path=os.path.join(save_dir, "log_llm_res.txt"),
        num_iter=i
    )

    if blueprint != "None":
        # Match detection results and real block names in robosuite env
        block_matches = match_block_names(blueprint["detected_blocks"], color_to_geom_names)
        print("Block matches: ", block_matches)

        # Get color information
        block_colors = {}
        for det in blueprint["detected_blocks"]:
            if "block_id" in det.keys():
                det_id = det["block_id"]
            else:
                det_id = det["id"]
            det_color = det["color"]
            block_colors[det_id] = det_color

        # Run the mid-level action planner
        table_center = env.sim.data.get_body_xpos("table")    # table center as the reference position
        action_planner = ActionPlanner(use_prim, table_center)
        assembly_order, new_block_positions = action_planner(blueprint)
            
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
    
        # Setup and run the low-level action executor
        executor = ActionExecutor(env, block_matches)
        executor(assembly_order, new_block_positions)

        # # Adjust pose of 'sideview' camera
        # cam_id = env.sim.model.camera_name2id("sideview")
        # env.sim.model.cam_pos[cam_id] = np.array([0.0, 0.9, 1.2])

        # Capture and save images from multiple views after assembling
        obs = env._get_observations()
        for cam_name in camera_names:
            # Ensure camera's RGB channel exists in observation and then get and save the RGB image
            rgb_cam_name = cam_name + "_image"
            if rgb_cam_name in obs.keys():
                rgb_img = obs[rgb_cam_name]  # Get image from the observation
                save_path = os.path.join(save_dir, f"after_assembling_{cam_name}_frame.png")
                save_cam_image(rgb_img, save_path)
    
    else:
        print("LLM returned 'None'!")
    
    # Reset the environment
    env.reset()

env.close()


