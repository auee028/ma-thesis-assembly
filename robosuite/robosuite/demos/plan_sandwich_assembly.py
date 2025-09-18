import numpy as np
import cv2
import os
from datetime import datetime
from collections import defaultdict

from robosuite.custom_utils.structure_planner import StructurePlanner, AssemblyLLM
from robosuite.custom_utils.assembly_spatial_graph import ObjectAssemblySpatialGraph
from robosuite.custom_utils.assembly_utils import *


def replace_names_with_ids(detected_objects, assembly_structure):
    # Create mapping dict of name -> id
    name_to_id = {obj["name"]: obj["id"] for obj in detected_objects}
    
    # Change assembly_structure
    replaced_structure = []
    for line in assembly_structure:
        parts = line.split()
        new_parts = []
        for part in parts:
            # Replace name by id when the mapping exists
            if part in name_to_id:
                new_parts.append(name_to_id[part])
            else:
                new_parts.append(part)
        replaced_structure.append(" ".join(new_parts))
    
    return replaced_structure

# Load config file
config_path = "../configs/assembly_configs_sandwich.json"
configs = read_file(config_path)

# Make a directory to save files
save_dir = os.path.join(configs["save_root"], f'captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}_{configs["exp_id"]}_{configs["task"]["target_query_structure"].replace(" ", "-")}')
configs["save_dir"] = save_dir
os.makedirs(save_dir, exist_ok=True)

# Save the current files
os.system(f"cp {config_path} {os.path.join(save_dir, os.path.basename(config_path))}")

target_rgb_path = configs["task"]["target_rgb_path"]
target_rgb_fname = os.path.basename(target_rgb_path)
configs["task"]["target_rgb_file"] = target_rgb_fname
os.system(f"cp {target_rgb_path} {os.path.join(save_dir, target_rgb_fname)}")

# Set up modules of LLM and Graph
if not configs["llm"]["use_llamaindex"]:
    use_prim = configs["relation"]["use_primitives"]
    structure_planner = StructurePlanner(configs, use_primitives=use_prim)
else:
    structure_planner = AssemblyLLM(configs)

# Iterate
for iter_id in range(configs["num_iter"]):
    # Get llm response
    print("### ", iter_id + 1)
    print("Query:", configs["task"]["target_query_structure"])
    blueprint = structure_planner(
        log_res_path=os.path.join(save_dir, "log_llm_res.txt"),
        num_iter=iter_id
    )
    
    detected_objects = blueprint["detected_objects"]

    if "assembly_structure" in blueprint.keys():
        assembly_structure = blueprint["assembly_structure"]
        print(f"Assembly structure: {assembly_structure}\n")

        replaced_structure = replace_names_with_ids(
            blueprint["detected_objects"], 
            blueprint["assembly_structure"]
        )
        print(f"ID-based structure: {replaced_structure}\n")
        
        g_obj = ObjectAssemblySpatialGraph()
        # spatial_graph, assembly_order = g_obj(assembly_structure)
        spatial_graph, assembly_order = g_obj(replaced_structure)

    else:
        print(blueprint.keys())

