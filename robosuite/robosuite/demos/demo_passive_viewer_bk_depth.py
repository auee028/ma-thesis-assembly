import numpy as np
import robosuite as suite
import cv2  # For saving images
import os
from datetime import datetime

# Create environment instance with offscreen rendering enabled
env = suite.make(
    env_name="Pyramid",  # Task: "Lift", "Stack", "Door", etc.
    robots="UR5e",  # Robot: "Sawyer", "Jaco", etc.
    has_renderer=False,  # No on-screen rendering
    has_offscreen_renderer=True,  # Enable offscreen rendering
    use_camera_obs=True,  # Enable camera observations
    camera_heights=512,
    camera_widths=512,
    camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_depths=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_segmentations='element',
    hard_reset=False,
    mujoco_passive_viewer=True
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

for i in range(3):  # Capture 3 frames
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

env.close()
