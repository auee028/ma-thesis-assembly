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

ENVNAME2SHAPES = {
    "Pyramid": ["cube"],
    "BigPyramid": ["cube"],
    "SmallTower": ["cube"],
    "Tower": ["cube"],
    "BigTower": ["cube"],
    "TwinTowers": ["cube"],
    "House": ["cube", "triangle"],
    "BigHouse": ["cube", "triangle"],
    "BigHousePillars": ["cube", "triangle"],
}

# Color mapping from name to RGBA
COLORNAME2RGBA = {
    "red": [1, 0.1, 0.1, 1],
    "orange": [1, 0.5, 0, 1],
    "yellow": [1, 0.95, 0.1, 1],
    "green": [0.1, 0.85, 0.2, 1],
    "blue": [0.1, 0.1, 0.85, 1],
    "purple": [0.65, 0, 0.72, 1],
}

# Color mapping from RGBA to name
RGBA2COLORNAME = {tuple(rgba): name for name, rgba in COLORNAME2RGBA.items()}

# LLM model costs 
MODEL_COSTS = {
    'gpt-4o': [2.50, 10.0],  # $2.50 / 1M input tokens, $10.00 / 1M output tokens
    'gpt-4o-2024-08-06': [2.50, 10.0],  # $2.50 / 1M input tokens, $10.00 / 1M output tokens'
    'gpt-4o-2024-11-20': [2.50, 10.0],  # $2.50 / 1M input tokens, $10.00 / 1M output tokens'
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
    Save RGB image captured from camera sensor in robosuite env
    """
    rgb_image = np.flipud(rgb_image)  # Flip vertically to match normal image orientation
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imwrite(save_path, rgb_image)  # Save the image
    print(f"Saved: {save_path}")

def save_cam_depth(depth_image, save_path):
    """
    Save depth image captured from camera sensor in robosuite env
    """
    depth_image = np.flipud(depth_image)  # Flip vertically to match normal image orientation
    depth_image_milli = (depth_image * 1000).astype(np.uint16)  # Scale to millimeters (common practice)
    cv2.imwrite(save_path, depth_image_milli)  # Save the image
    print(f"Saved: {save_path}")
    
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)  # Convert to uint8 for proper image encoding
    save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace("_depth", "_depth_vis"))
    cv2.imwrite(save_path, depth_uint8)  # Save the image
    print(f"Saved: {save_path}")

def save_cam_mask(seg_image, block_geom_ids, save_path):
    """
    Save segmentation mask of blocks in robosuite env
    """
    seg_image = np.flipud(seg_image)  # Flip vertically to match normal image orientation
    blocks_seg_img = np.zeros_like(seg_image)
    for block_id in block_geom_ids:
        blocks_seg_img[seg_image == block_id] = 1
    cv2.imwrite(save_path, blocks_seg_img)  # Save the image
    print(f"Saved: {save_path}")
    
    blocks_seg_img = blocks_seg_img * 255
    save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace("_mask", "_mask_vis"))
    cv2.imwrite(save_path, blocks_seg_img)  # Save the image
    print(f"Saved: {save_path}")

def save_fused_rgbd(rgb_image, depth_image, save_path):
    """
    Save fused RGB-D in robosuite env
    """
    depth_image = np.flipud(depth_image)  # Flip vertically to match normal image orientation
    depth_image_milli = (depth_image * 1000).astype(np.uint16)  # Scale to millimeters (common practice)
    save_milli_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace("_rgbd", "_depth_milli"))
    cv2.imwrite(save_milli_path, depth_image_milli)  # Save the image
    print(f"Saved: {save_milli_path}")
    
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)  # Convert to uint8 for proper image encoding
    save_vis_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace("_rgbd", "_depth_vis"))
    cv2.imwrite(save_vis_path, depth_uint8)  # Save the image
    print(f"Saved: {save_vis_path}")
    
    rgbd_image = np.dstack((rgb_image, depth_image_milli))    # Mixed RGBA data (depth hidden in alpha channel)
    rgbd_image = rgbd_image[:, :, [2, 1, 0, 3]]  # Convert RGB channels BACK to BGR for OpenCV before saving
    cv2.imwrite(save_path, rgbd_image)  # Save the image
    print(f"Saved: {save_path}")

def save_fused_rgbseg(rgb_image, seg_image, block_geom_ids, save_path):
    """
    Save segmentation of blocks in robosuite env
    """
    rgb_image = np.flipud(rgb_image)  # Flip vertically to match normal image orientation
    seg_image = np.flipud(seg_image)  # Flip vertically to match normal image orientation
    blocks_seg_img = np.zeros_like(seg_image).astype(np.uint8)  # from 32-bit to 8-bit int
    for block_id in block_geom_ids:
        blocks_seg_img[seg_image == block_id] = 1
    rgb_seg = rgb_image * blocks_seg_img
    rgb_seg = cv2.cvtColor(rgb_seg, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imwrite(save_path, rgb_seg)  # Save the image
    print(f"Saved: {save_path}")
    
    blocks_seg_img = blocks_seg_img * 255
    save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace("_rgbseg", "_mask_vis"))
    cv2.imwrite(save_path, blocks_seg_img)  # Save the image
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
        rows = f.readlines()
        last_call = rows[-1].strip()
    if len(rows) > 1:
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


def compute_positions_from_distances(dist_relationships):
    block_positions = {}
    
    for relation in dist_relationships:
        src_block = relation["from"]
        tgt_block = relation["to"]
        center_distance = relation["center_distance"]
        relative_dist = np.array([center_distance["x"], center_distance["y"], center_distance["z"]])
        
        if src_block not in block_positions.keys():
            block_positions[src_block] = np.array([0., 0., 0.])
        
        base_position = block_positions[src_block]
        block_positions[tgt_block] = base_position + relative_dist
    
    assembly_order = [*(block_positions.keys())]

    return assembly_order, block_positions

def compute_positions_from_instructions(spatial_graph, assembly_order):
    """
    compute the 3d position of each block given instructions in natural language
    """
    block_size = np.array([0.04, 0.04, 0.04])  # assuming cubic block
    # block_size = 0.04  # assuming uniform block length
    
    base_pos = np.array([0.0, 0.0, 0.0])
    
    block_positions = {}
    base_blocks = []
    
    for i, block in enumerate(assembly_order):
        relationships = spatial_graph.get(block, [])
        
        for relationship in relationships:
            relation = relationship[0]
            supporters = relationship[1:]
            
            # Move blocks according to the relation to its supporter
            if relation == "base":
                if len(base_blocks) == 0:
                    new_pos = base_pos
                else:
                    new_pos = base_blocks[-1][1] + np.array([block_size[0], 0.0, 0.0])
                base_blocks.append(tuple([block, new_pos]))
            
            elif relation == "right":
                supp_pos = block_positions[supporters[0]]
                new_pos = supp_pos + np.array([block_size[0], 0.0, 0.0])
                
            elif relation == "left":
                supp_pos = block_positions[supporters[0]]
                new_pos = supp_pos + np.array([-block_size[0], 0.0, 0.0])
                
            elif relation == "front":
                supp_pos = block_positions[supporters[0]]
                new_pos = supp_pos + np.array([0.0, block_size[1], 0.0])
                
            elif relation == "behind":
                supp_pos = block_positions[supporters[0]]
                new_pos = supp_pos + np.array([0.0, -block_size[1], 0.0])
                
            elif relation == "top" and len(supporters) == 1:
                # Place directly on top
                supp_pos = block_positions[supporters[0]]
                new_pos = supp_pos + np.array([0.0, 0.0, block_size[2]])
                
            elif len(supporters) > 1:
                assert relation == "top"
                # Center between two supporters
                supp1, supp2 = block_positions[supporters[0]], block_positions[supporters[1]]
                avg_pos = (supp1 + supp2) / 2
                avg_pos[2] += block_size[2]
                new_pos = avg_pos
            
            block_positions[block] = new_pos
    
    return block_positions


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

def plot_triangle(ax, position, color, length=0.04, depth=0.15, height=0.05, alpha=0.4, rotate=True):
    """
    Plot a triangular prism centered at 'position'.
    The triangle lies in the YZ plane and is extruded along the X-axis.
    Set `rotate=True` to rotate the whole prism 90° around the Z-axis.
    """
    x, y, z = position
    half_len = length / 2
    half_depth = depth / 2
    half_height = height / 2

    # Local triangle in YZ plane, extruded along X
    # Start with full 3D coordinates
    local_tri = np.array([
        [-half_len, -half_depth, -half_height],
        [-half_len,  half_depth, -half_height],
        [-half_len, 0.0,         half_height],
        [ half_len, -half_depth, -half_height],
        [ half_len,  half_depth, -half_height],
        [ half_len, 0.0,         half_height],
    ])

    if rotate:
        # Rotation around Z-axis by 90 degrees
        Rz = np.array([
            [0, -1, 0],  # x -> -y
            [1,  0, 0],  # y -> x
            [0,  0, 1]   # z -> z (no change)
        ])
        local_tri = local_tri @ Rz.T
        
        local_tri[:, 1] -= half_depth / 2  # Due to unexpected half of depth more in y axis
        local_tri[:, 2] -= (0.05 - 0.04) / 2  # Due to different heights of cube and triangle

    # Translate to global position
    global_tri = local_tri + np.array([x, y, z])

    # Define faces
    faces = {
        "front": [0, 1, 2],
        "back":  [5, 4, 3],
        "side1": [0, 3, 4, 1],
        "side2": [1, 4, 5, 2],
        "side3": [2, 5, 3, 0]
    }

    # Plot front and back triangles
    def plot_face(indices, is_triangle=False):
        verts = global_tri[indices]
        if is_triangle:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], [verts[i, 2] for i in range(3)],
                            triangles=[[0, 1, 2]], color=color, alpha=alpha, edgecolor="k")
        else:
            xx = np.array([[verts[0][0], verts[1][0]], [verts[3][0], verts[2][0]]])
            yy = np.array([[verts[0][1], verts[1][1]], [verts[3][1], verts[2][1]]])
            zz = np.array([[verts[0][2], verts[1][2]], [verts[3][2], verts[2][2]]])
            ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, edgecolor="k")

    plot_face(faces["front"], is_triangle=True)
    plot_face(faces["back"], is_triangle=True)
    plot_face(faces["side1"])
    plot_face(faces["side2"])
    plot_face(faces["side3"])

def set_axes_equal(ax):
    '''Make the 3D plot have equal aspect ratio.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def visualize_structure(assembly_order, block_positions, block_matches, block_colors, query_structure, save_file_path=None, block_length=0.04, alpha=0.4, zoom=1.25, axis_on=True):
    """
    Plot and visualize the resulting structure
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(block_positions) == 0:
        print("Warning: Empty positions list. Visualizing without any blocks.")
    else:
        # Plot each block
        for i, block_id in enumerate(assembly_order):
            block_pos = block_positions.get(block_id, '')
            geom_name = block_matches.get(block_id, '')
            block_color = block_colors.get(block_id, '')
            
            if not geom_name:
                print(f"{bcolors.RED}Warning: No matched geom_name left for '{block_id}' (color: {block_color}){bcolors.RESET}")
                continue
            else:
                # Plot the block
                if "cube" in geom_name:
                    plot_cube(ax, block_pos, block_color, block_length, alpha=alpha)
                elif "triangle" in geom_name:
                    plot_triangle(ax, block_pos, block_color, alpha=alpha)
                ax.text(*(block_pos+[0,0,-0.005]), block_id, fontsize=16, ha="center")
                # ax.text(*block_pos, block_id, fontsize=16, ha="center")
                
                # Plot connection between blocks
                if i < len(assembly_order) - 1:
                    from_block = block_id
                    to_block = assembly_order[i+1]
                    pos_from = block_positions.get(from_block)
                    pos_to = block_positions.get(to_block)
                    ax.plot(
                        [pos_from[0], pos_to[0]], [pos_from[1], pos_to[1]], [pos_from[2], pos_to[2]],
                        color="black", linestyle="--", linewidth=2.0
                    )
        
        # Collect all positions to determine axis limits
        positions = np.array([position for position in block_positions.values()])
        margin = block_length * 1.0  # 0.7

        xmin_limit = positions[:,0].min() - margin
        ymin_limit = positions[:,1].min() - margin
        zmin_limit = positions[:,2].min() - margin
        xmax_limit = positions[:,0].max() + margin
        ymax_limit = positions[:,1].max() + margin
        zmax_limit = positions[:,2].max() + margin

        # Set the same range for each axis to maintain equal scaling
        ax.set_xlim([xmin_limit, xmax_limit])
        ax.set_ylim([ymin_limit, ymax_limit])
        ax.set_zlim([zmin_limit, zmax_limit])

        # Set equal aspect ratio for 3D axes
        set_axes_equal(ax)

    # Labels and view adjustments
    ax.set_xlabel("X", labelpad=-8, fontsize=14)
    ax.set_ylabel("Y", labelpad=-10, fontsize=14)
    ax.set_zlabel("Z", labelpad=-10, fontsize=14)
    ax.set_title(f"Query structure: {query_structure}", fontsize=20)

    # ax.view_init(20, 170)   # rotate
    ax.view_init(15, 100)   # rotate
    # ax.view_init(15, 280)   # rotate
    ax.set_box_aspect((1, 1, 1), zoom=zoom)  # zoom

    if axis_on:
        plt.axis('on')
    else:
        plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if save_file_path is not None:
        fig.savefig(save_file_path)
    # plt.show()


