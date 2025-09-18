import numpy as np
import robosuite as suite
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map
import cv2  # For saving images
import os
from datetime import datetime
import open3d as o3d


# Create environment instance with offscreen rendering enabled
env = suite.make(
    env_name="Pyramid",  # Task: "Lift", "Stack", "Door", etc.
    robots="UR5e",  # Robot: "Sawyer", "Jaco", etc.
    has_renderer=False,  # No on-screen rendering
    has_offscreen_renderer=True,  # Enable offscreen rendering
    use_camera_obs=True,  # Enable camera observations
    camera_heights=512,
    camera_widths=512,
    # camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    # camera_depths=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_names=['agentview', 'robot0_eye_in_hand'],
    camera_depths=['agentview', 'robot0_eye_in_hand'],
    camera_segmentations='element',
    hard_reset=False,
    mujoco_passive_viewer=False
)

# Access the geometries' names and ids
geom_names = env.sim.model.geom_names
block_names = [geom_name for geom_name in geom_names
                if ('cube' in geom_name) and ('_vis' in geom_name)]
block_ids = [env.sim.model.geom_name2id(block_name) for block_name in block_names]
print(block_ids, block_names)

# Define camera views to capture
camera_names = env.sim.model.camera_names
# print(camera_names)  # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')

# Create a directory to save images
save_dir = f'/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(save_dir, exist_ok=True)

# Reset the environment
env.reset()
    
for i in range(1):  # Capture 1 frames
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)
    print(obs.keys())

    # Capture and save images from multiple views
    for cam_name in camera_names:
        # Ensure camera's RGB channel exists in observation and then get and save the RGB image
        rgb_cam_name = cam_name + "_image"
        if rgb_cam_name in obs.keys():
            rgb_img = obs[rgb_cam_name]  # Get image from the observation
            rgb_img = np.flipud(rgb_img)  # Flip vertically to match normal image orientation
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_rgb.png")
            cv2.imwrite(save_path, rgb_img)  # Save the image
            print(f"Saved: {save_path}")
            
        # Ensure camera's depth channel exists in observation and then get and save the depth image
        depth_cam_name = cam_name + "_depth"
        if depth_cam_name in obs.keys():
            depth_img = obs[depth_cam_name]  # Get image from the observation
            depth_img = np.flipud(depth_img)  # Flip vertically to match normal image orientation
            
            # Convert MuJoCo's depth to real-world metric values
            depth_img = get_real_depth_map(env.sim, depth_img)
            print(depth_img.min(), depth_img.max())
            
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_depth.npy")
            np.save(save_path, depth_img)
            print(f"Saved: {save_path}")
            
            depth_img_scaled = (depth_img * 1000).astype(np.uint16)  # Scale to millimeters (common practice)
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_depth_milli.png")
            cv2.imwrite(save_path, depth_img_scaled)
            print(f"Saved: {save_path}")
            
            """
            # Clip and normalize the depth values in the range [0.0, 0.6] (30 cm ~ 60 cm)
            clip_min = 0.98
            clip_max = 1.0
            #depth_img = np.where((depth_img >= clip_min) & (depth_img <= clip_max), depth_img, np.nan)  # NaN when out of range
            #depth_img = np.clip(depth_img, clip_min, clip_max)
            depth_img_scaled = (depth_img - clip_min) / (clip_max - clip_min)  # Normalize to [clip_min, clip_max]
            depth_img_scaled_encoded = (depth_img_scaled * 65535.0).astype(np.uint16)
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_depth_scaled.png")
            cv2.imwrite(save_path, depth_img_scaled_encoded)
            print(f"Saved: {save_path}")
            """
            # # Clip and the depth values in the range [clip_min, clip_max] (in meters)
            # clip_min = 0.3
            # clip_max = 1.0
            # depth_bg = 1.0
            # depth_img = np.where((depth_img >= clip_min) & (depth_img <= clip_max), depth_img, depth_bg)
            # depth_img_normalized = (depth_img - clip_min) / (clip_max - clip_min)
            # depth_img_uint8 = (depth_img_normalized * 255).astype(np.uint8)
            # Normalize the depth values to [0, 255] for visualization
            depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            # Convert to uint8 for proper image encoding
            depth_img_uint8 = depth_img_normalized.astype(np.uint8)
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_depth_vis.png")
            # cv2.imwrite(save_path, depth_img_encoded, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # value 0 for no compression
            cv2.imwrite(save_path, depth_img_uint8)
            print(f"Saved: {save_path}")
        
        # Ensure camera's segmentation exists in observation and then get and save the depth image
        seg_cam_name = cam_name + "_segmentation_element"
        if seg_cam_name in obs.keys():
            seg_img = obs[seg_cam_name]  # Get image from the observation
            seg_img = np.flipud(seg_img)  # Flip vertically to match normal image orientation
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_seg.png")
            cv2.imwrite(save_path, seg_img)  # Save the image
            print(f"Saved: {save_path}")
        
            # Save block segmentation image
            blocks_seg_img = np.zeros_like(seg_img)
            for block_id in block_ids:
                blocks_seg_img[seg_img == block_id] = 1
                # print(depth_img[seg_img == block_id].min(), depth_img[seg_img == block_id].max())
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_seg_blocks_mask.png")
            cv2.imwrite(save_path, blocks_seg_img * 255)  # Save the scaled image
            print(f"Saved: {save_path}")
            
            blocks_depth_img = depth_img * blocks_seg_img
            blocks_depth_img = (blocks_depth_img * 65535.0).astype(np.uint16)
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_depth_blocks.png")
            cv2.imwrite(save_path, blocks_depth_img)  # Save the scaled image
            print(f"Saved: {save_path}")
            
        # Generate and Save Point Cloud
        if (rgb_cam_name in obs.keys()) and (depth_cam_name in obs.keys()):
            # rgb_img = np.flipud(rgb_img)
            # depth_img = np.flipud(depth_img)
            
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            depth_img = depth_img.squeeze()
            
            # Camera-to-world transformation matrix
            cam_h, cam_w = depth_img.shape
            camera_transform = get_camera_transform_matrix(env.sim, cam_name, cam_h, cam_w)

            # Back-Projection to 3D Space
            u, v = np.meshgrid(np.arange(cam_w), np.arange(cam_h))
            x = (u - camera_transform[0, 2]) * depth_img / camera_transform[0, 0]
            y = (v - camera_transform[1, 2]) * depth_img / camera_transform[1, 1]
            z = depth_img

            # Stack points into shape (N, 3)
            points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

            # Convert points to homogeneous coordinates
            points = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

            # Transform Points to World Coordinates
            points = (np.linalg.inv(camera_transform) @ points.T).T[:, :3]
            
            # Reshape RGB image to match point cloud
            colors = rgb_img.reshape(-1, 3) / 255.0

            # Create and Save Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}_pointcloud.ply")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"Saved Point Cloud: {save_path}")

env.close()
