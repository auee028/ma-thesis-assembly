import numpy as np
import cv2
import os
from datetime import datetime
from collections import defaultdict

import robosuite
from robosuite.custom_utils.assembly_llm import AssemblyLLM, AssemblyLLMOpenAI
from robosuite.custom_utils.assembly_spatial_graph import ObjectAssemblySpatialGraph # AssemblySpatialGraph
from robosuite.custom_utils.assembly_utils import *


# Load config file
# config_path = "../configs/thesis_exp_configs.json"
config_path = "../configs/thesis_exp_configs_sandwich.json"
configs = read_file(config_path)

# Make a directory to save files
save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}_thesis_{configs["exp_id"]}_{configs["task"]["target_query_structure"].replace(" ", "-")}')
# save_dir += f'_topp{format(configs["llm"]["model_config"]["top_p"], ".0e").replace("0e", "e")}'  # ex) "0.2" -> "2e-1"
os.mkdir(save_dir)

# Save the current files
os.system(f"cp ../custom_utils/assembly_llm.py {os.path.join(save_dir, 'assembly_llm.py')}")
os.system(f"cp {config_path} {os.path.join(save_dir, os.path.basename(config_path))}")

# mode = configs["relation"]["dist_mode"]
mode = False
if mode is False:    # When applying spatial relation primitives
    system_prompt = configs["relation"]["prim_prompt_files"]["system_prompt"]
    os.system(f"cp {system_prompt} {os.path.join(save_dir, os.path.basename(system_prompt))}")

    examples_prompt = configs["relation"]["prim_prompt_files"]["examples_prompt"]
    os.system(f"cp {examples_prompt} {os.path.join(save_dir, os.path.basename(examples_prompt))}")
else:    # When applying 3D distance relationships
    dist_examples_prompt = configs["relation"]["dist_prompt_files"]["examples_prompt"]
    os.system(f"cp {dist_examples_prompt} {os.path.join(save_dir, os.path.basename(dist_examples_prompt))}")

    dist_example_rgb_paths = configs["relation"]["dist_prompt_files"]["example_rgb_paths"]
    if dist_example_rgb_paths:
        os.system(f"cp {dist_example_rgb_paths} {os.path.join(save_dir, os.path.basename(dist_example_rgb_paths))}")
        
    dist_example_depth_paths = configs["relation"]["dist_prompt_files"]["example_depth_paths"]
    if dist_example_depth_paths:
        os.system(f"cp {dist_example_depth_paths} {os.path.join(save_dir, os.path.basename(dist_example_depth_paths))}")
        
    dist_example_seg_paths = configs["relation"]["dist_prompt_files"]["example_seg_paths"]
    if dist_example_seg_paths:
        os.system(f"cp {dist_example_seg_paths} {os.path.join(save_dir, os.path.basename(dist_example_seg_paths))}")


# Set up modules of LLM and Graph
# llm = AssemblyLLM(configs)
# llm = AssemblyLLMOpenAI(configs)
llm = AssemblyLLMOpenAI(configs, distance_relations=mode)

# Iterate
for iter_id in range(configs["num_iter"])[:1]:
    # Get llm response
    print("### ", iter_id + 1)
    print("Query:", configs["task"]["target_query_structure"])
    content = llm(log_res_path=os.path.join(save_dir, "log_llm_res.txt"), num_iter=iter_id)    # returns 'content' of response from LLM
    # detected_blocks = content["detected_blocks"]
    detected_objects = content["detected_objects"]

    if "assembly_structure" in content.keys():
        assembly_structure = content["assembly_structure"]
        print(f"Assembly structure: {assembly_structure}\n")
        
        g_obj = ObjectAssemblySpatialGraph()
        spatial_graph, assembly_order = g_obj(assembly_structure)

    else:
        print(content.keys())
