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
    camera_names=["frontview", "birdview", "agentview", "sideview"],
    hard_reset=False,
    mujoco_passive_viewer=True
)

# Define camera views to capture
camera_names = env.sim.model.camera_names

# Create a directory to save images
save_dir = f'/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_{datetime.today().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(save_dir, exist_ok=True)

# Reset the environment
env.reset()

for i in range(3):  # Capture 3 frames
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)
    print(obs.keys())

    """
    # Capture and save images from multiple views
    for cam_name in camera_names:
        cam_name = cam_name + "_image"
        if cam_name in obs.keys():  # Ensure camera exists in observation
            img = obs[cam_name]  # Get image from the observation
            img = np.flipud(img)  # Flip vertically to match normal image orientation
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            save_path = os.path.join(save_dir, f"{cam_name}_frame_{i}.png")
            cv2.imwrite(save_path, img)  # Save the image
            print(f"Saved: {save_path}")
    """
    # Capture and save images from multiple views
    for cam_name in camera_names:
        # Capture RGB image
        rgb_img = env.sim.render(camera_name=cam_name, width=512, height=512, depth=False)

        # Capture Depth image
        _, depth_img = env.sim.render(camera_name=cam_name, width=512, height=512, depth=True)

        # Convert RGB to OpenCV format and save
        rgb_img = np.flipud(rgb_img)  # Flip vertically
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        cv2.imwrite(os.path.join(save_dir, f"{cam_name}_frame_{i}_rgb.png"), rgb_img)

        # Convert Depth to 8-bit grayscale for visualization
        depth_img = np.flipud(depth_img)  # Flip vertically
        # Method 1. Normalizing
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)  # Normalize to [0, 1]
        depth_img = (depth_img * 255).astype(np.uint8)  # Convert to 8-bit grayscale
        # Method 2. Convert meters to millimeters
        # depth_img = depth_img * 1000    # Convert the depth in meters (e.g., 0.4 for 40cm) to millimeters (400 for 40 cm)
        # depth_img = depth_img.astype(np.uint16)  # Convert to uint16 (uint16 (16-bit) can represent values 0 to 65,535)
        # print(depth_img.min(), depth_img.max())
        cv2.imwrite(os.path.join(save_dir, f"{cam_name}_frame_{i}_depth.png"), depth_img)

        print(f"Saved: {cam_name}_frame_{i}_rgb.png and {cam_name}_frame_{i}_depth.png")

    

env.close()
