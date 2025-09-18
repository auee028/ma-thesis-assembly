import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import cv2
from robosuite.utils.transform_utils import mat2quat

from robosuite.custom_utils.voxposer_utils import bcolors


BASE_BLOCK_ALIAS = ["at the base", "on the table", "at the bottom"]

# Color mapping from name to RGBA
COLOR_NAME2RGBA = {
    "red": [1, 0.1, 0.1, 1],
    "orange": [1, 0.4, 0, 1],
    "yellow": [1, 0.95, 0.1, 1],
    "green": [0.1, 0.85, 0.2, 1],
    "blue": [0.1, 0.1, 0.85, 1],
    "purple": [0.85, 0.1, 0.85, 1],
}

# Color mapping from RGBA to name
COLOR_RGBA2NAME = {tuple(rgba): name for name, rgba in COLOR_NAME2RGBA.items()}

# LLM model costs 
MODEL_COSTS = {
    'gpt-4o': [2.50, 10.0],  # $2.50 / 1M input tokens, $10.00 / 1M output tokens
    'gpt-4o-2024-08-06': [2.50, 10.0],  # $2.50 / 1M input tokens, $10.00 / 1M output tokens'
    'mistral-small-latest': [0, 0],  # free model from mistral
    'pixtral-12b-2409': [0, 0],  # free model from mistral
    'gemini-2.0-flash': [0, 0],  # free model from google
}


def get_color_name(rgba, color_map, tolerance=0.01):
    """
    Find the closest color name for a given RGBA
    """
    # Compare with some tolerance for floating point values
    for color_rgba, name in color_map.items():
        if np.allclose(rgba, color_rgba, atol=tolerance):
            return name
    return None


def read_file(file_path):
    """
    Read a file like .txt and .json
    """    
    ext = file_path.split(os.pathsep)[-1].split('.')[-1]
    if ext == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(file_path, 'r') as f:
            contents = f.read().strip()
    return contents


def string_to_json(json_string):
    """
    Convert a json string to json format
    """
    json_string = json_string.replace("```json", "").replace("```", "").strip()
    return json.loads(json_string)


def encode_image(image_path):
    """
    Encode an image into base64 format
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def match_block_names(detected_blocks, color_to_geom_names):
    """
    Match detected blocks with environment observations
    """
    block_matches = {}
    for i, det in enumerate(detected_blocks):
        if "block_id" in det.keys():
            block_id = det["block_id"]
        else:
            block_id = det["id"]
        block_color = det["color"]
        
        if block_color in color_to_geom_names and color_to_geom_names[block_color]:
            # Pop the first available geom_name (removes it from the list)
            geom_name = color_to_geom_names[block_color].pop(0)
            block_matches[block_id] = geom_name.replace('_g0_vis', '')
        elif block_color == "magenta":
            # TODO: How to deal with color alias such as 'purple' and 'magenta'
            block_color = "purple"
            geom_name = color_to_geom_names[block_color].pop(0)
            block_matches[block_id] = geom_name.replace('_g0_vis', '')
        else:
            print(f"{bcolors.RED}Warning: No unassigned geom_name left for {block_id} (color: {block_color}){bcolors.RESET}")
    
    return block_matches


def save_cam_image(rgb_image, save_path):
    """
    Save image captured from camera sensor in robosuite env
    """
    rgb_image = np.flipud(rgb_image)  # Flip vertically to match normal image orientation
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imwrite(save_path, rgb_image)  # Save the image
    print(f"Saved: {save_path}")


def log_token_usage(ts, model, input_tokens, output_tokens, total_tokens, verbose=True, token_log_dir='../logs'):
    """
    Log LLM tokens usage
    """
    now = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")
    year_month = datetime.utcfromtimestamp(ts).strftime("%Y-%m")
    
    token_log_path = os.path.join(token_log_dir, f"log_openai_token_usage_{year_month}.txt")

    if not os.path.exists(token_log_path):
        with open(token_log_path, 'w') as f:
            f.write(f'date_time\tmodel\tinput_tokens(input_costs)\toutput_tokens(output_costs)\ttotal_tokens(total_costs)\taccumulated_cost[dolar]\n')
    
    with open(token_log_path, 'r') as f:
        rows = f.read()
        last_call = rows[-1].strip()
    if last_call:
        accumulated_cost = float(last_call.split('\t')[-1])
    else:
        accumulated_cost = 0.0
    
    input_cost = MODEL_COSTS[model][0] / 1000000 * input_tokens
    output_cost = MODEL_COSTS[model][1] / 1000000 * output_tokens
    summed_cost = (input_cost + output_cost)
    accumulated_cost += summed_cost

    logging = f'{now}\t{model}\t{input_tokens}({input_cost})\t{output_tokens}({output_cost})\t{total_tokens}({summed_cost})\t{accumulated_cost}\n'
    
    if verbose:
        print('Logging: ' + logging + '')

    with open(token_log_path, 'a') as f:
        f.write(logging)


def plot_cube(ax, position, color, length, alpha=0.4):
    """
    Function to plot a cube centered at a given position
    """
    # Calculate the edges of the cube
    x, y, z = position
    l = length / 2  # Half-length to make the cube centered at the position

    # Define each face of the cube using the center position and half-length
    # Each face needs to be defined by its corners in 2D grids

    # Bottom face (z - l)
    xx, yy = np.meshgrid([x - l, x + l], [y - l, y + l])
    ax.plot_surface(xx, yy, (z - l) * np.ones_like(xx), color=color, edgecolor="k", alpha=alpha)

    # Top face (z + l)
    ax.plot_surface(xx, yy, (z + l) * np.ones_like(xx), color=color, edgecolor="k", alpha=alpha)

    # Front face (y - l)
    xx, zz = np.meshgrid([x - l, x + l], [z - l, z + l])
    ax.plot_surface(xx, (y - l) * np.ones_like(xx), zz, color=color, edgecolor="k", alpha=alpha)

    # Back face (y + l)
    ax.plot_surface(xx, (y + l) * np.ones_like(xx), zz, color=color, edgecolor="k", alpha=alpha)

    # Left face (x - l)
    yy, zz = np.meshgrid([y - l, y + l], [z - l, z + l])
    ax.plot_surface((x - l) * np.ones_like(yy), yy, zz, color=color, edgecolor="k", alpha=alpha)

    # Right face (x + l)
    ax.plot_surface((x + l) * np.ones_like(yy), yy, zz, color=color, edgecolor="k", alpha=alpha)


def visualize_structure(structure_dict, alpha=0.4, zoom=1.25, axis_on=True, save_fig=True):
    """
    Plot and visualize the resulting structure
    """
    input_structure = structure_dict['input_structure']
    detected_blocks = structure_dict['detected_blocks']
    block_length = structure_dict['block_length']
    structure_relations = structure_dict['structure_relations']

    # if "id" in detected_blocks[0].keys():
    #     block_properties = {det["id"]: {"color": det["color"]} for det in detected_blocks}
    # elif "block_id" in detected_blocks[0].keys():
    #     block_properties = {det["block_id"]: {"color": det["color"]} for det in detected_blocks}
    block_properties = {det["block_id"]: {"color": det["color"]} for det in detected_blocks}

    for relation in structure_relations:
        src_block = relation["from"]
        tgt_block = relation["to"]
        center_distance = relation["center_distance"]
        
        if "position" not in block_properties[src_block].keys():
            block_properties[src_block]["position"] = np.array([0., 0., 0.])
        
        base_position = block_properties[src_block]["position"]
        relative_dist = np.array([center_distance["x"], center_distance["y"], center_distance["z"]])
        block_properties[tgt_block]["position"] = base_position + relative_dist

    # Change structure relations according to the modified structure
    if 'modified_relations' in structure_dict.keys():
        for modified_relation in structure_dict['modified_relations']:
            src_block = modified_relation["from"]
            tgt_block = modified_relation["to"]
            center_distance = modified_relation["center_distance"]
            
            base_position = block_properties[src_block]["position"]
            relative_dist = np.array([center_distance["x"], center_distance["y"], center_distance["z"]])
            block_properties[tgt_block]["position"] = base_position + relative_dist
    
    # Remove the block which is not used for any relationships
    for block_id in list(block_properties.keys()):
        if "position" not in block_properties[block_id].keys():
            block_properties.pop(block_id)

    save_file_path = f'output/vis_3d_{input_structure}_{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.png' # f'/home/user/parkj0/VoxPoser/src/output/vis_3d_{input_structure}_{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    # print(save_file_path)
    
    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each block as a cube
    for block_id, properties in block_properties.items():
        plot_cube(ax, properties["position"], properties["color"], block_length, alpha=alpha)
        if block_id == 'block0':
            ax.text(*properties["position"]+[0,0,-0.005], block_id, fontsize=16, ha="center")
        else:
            ax.text(*properties["position"], block_id, fontsize=16, ha="center")

    # Plot connections based on structure_relations
    for relation in structure_relations:
        from_block = relation["from"]
        to_block = relation["to"]
        pos_from = block_properties[from_block]["position"]
        pos_to = block_properties[to_block]["position"]
        ax.plot(
            [pos_from[0], pos_to[0]], [pos_from[1], pos_to[1]], [pos_from[2], pos_to[2]],
            color="black", linestyle="--", linewidth=2.0
        )
    
    # Collect all positions to determine axis limits
    positions = np.array([props["position"] for props in block_properties.values()])
    min_limit = positions.min() - block_length/10
    max_limit = positions.max() + block_length/10

    # Set the same range for each axis to maintain equal scaling
    ax.set_xlim([min_limit - block_length/2, max_limit - block_length/2])
    ax.set_ylim([min_limit - block_length/2, max_limit - block_length/2])
    ax.set_zlim([min_limit + block_length/2, max_limit + block_length/2])

    # Labels and view adjustments
    ax.set_xlabel("X", labelpad=-8, fontsize=14)
    ax.set_ylabel("Y", labelpad=-10, fontsize=14)
    ax.set_zlabel("Z", labelpad=-10, fontsize=14)
    ax.set_title(f"Query structure: {input_structure} ({os.path.basename(save_file_path)})", fontsize=20)

    # ax.view_init(20, 170)   # rotate
    ax.view_init(15, 10)   # rotate
    ax.set_box_aspect((1, 1, 1), zoom=zoom)  # zoom

    if axis_on:
        plt.axis('on')
    else:
        plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save_fig:
        fig.savefig(save_file_path)
        # fig.savefig(f'/home/user/parkj0/VoxPoser/src/output/vis_3d_{input_structure}.png')
    plt.show()

