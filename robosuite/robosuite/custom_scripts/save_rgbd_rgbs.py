import os
import numpy as np
import cv2


ex_root = "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer"
ex_dirs = [
    "captured_images_20250507-155702_ex1",
    "captured_images_20250507-155752_ex2",
    "captured_images_20250507-160214_ex3",
    "captured_images_20250516-002905_ex4",
]

for ex_dir in ex_dirs:
    rgb_fname = "agentview_frame_0_rgb.png"
    depth_fname = "agentview_frame_0_depth_milli.png"
    mask_fname = "agentview_frame_0_seg_blocks_mask.png"
    
    rgb_path = os.path.join(ex_root, ex_dir, rgb_fname)
    depth_path = os.path.join(ex_root, ex_dir, depth_fname)
    mask_path = os.path.join(ex_root, ex_dir, mask_fname)
    
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    print("RGB Shape: ", rgb.shape)
    
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # Ensures 16-bit loading
    print("Depth Shape: ", depth.shape)
    
    # Convert depth to a heatmap (normalized 0-255)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    print("Depth Heatmap Shape: ", depth_heatmap.shape)
    heatmap_bgr = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)  # Convert BACK to BGR before saving
    cv2.imwrite(os.path.join(ex_root, ex_dir, depth_fname.replace("depth_milli", "depth_heatmap")), heatmap_bgr)
    # Clip background
    depth_min = 620
    depth_max = 1050
    depth_clipped = np.clip(depth, depth_min, depth_max)
    depth_clipped_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_clipped_heatmap = cv2.applyColorMap(depth_clipped_normalized, cv2.COLORMAP_JET)
    print("Clipped Depth Heatmap Shape: ", depth_clipped_heatmap.shape)
    depth_clipped_heatmap_bgr = cv2.cvtColor(depth_clipped_heatmap, cv2.COLOR_RGB2BGR)  # Convert BACK to BGR before saving
    cv2.imwrite(os.path.join(ex_root, ex_dir, depth_fname.replace("depth_milli", "depth_clipped_heatmap")), depth_clipped_heatmap_bgr)
    
    # RGB + Depth: [ [ [R, G, B, Depth], ... ] ]
    rgbd_channelstacked = np.dstack((rgb, depth))    # Mixed RGBA data (depth hidden in alpha channel)
    print("RGBD Shape (channel-stacked): ", rgbd_channelstacked.shape)
    rgbd_channelstacked = rgbd_channelstacked[:, :, [2, 1, 0, 3]]  # Convert RGB part BACK to BGR before saving
    cv2.imwrite(os.path.join(ex_root, ex_dir, rgb_fname.replace("rgb", "rgbd_channelstacked")), rgbd_channelstacked)
    
    # RGB + Depth: [RGB Image] | [Depth Heatmap]
    rgbd_sidebyside = np.hstack((rgb, depth_heatmap))  # Horizontal stack 
    print("RGBD Shape (side-by-side): ", rgbd_sidebyside.shape)
    rgbd_sidebyside = cv2.cvtColor(rgbd_sidebyside, cv2.COLOR_RGB2BGR)  # Convert BACK to BGR before saving
    cv2.imwrite(os.path.join(ex_root, ex_dir, rgb_fname.replace("rgb", "rgbd_sidebyside")), rgbd_sidebyside)
    
    # RGB + Depth: [ [ [R, G, B, HeatmapR, HeatmapG, HeatmapB], ... ] ]
    # Cannot be saved as a PNG or JPEG because standard image formats (like PNG/JPEG) do not support 6 channels.
    # They typically support:
    #    - 1 channel (grayscale)
    #    - 3 channels (RGB)
    #    - 4 channels (RGBA, with transparency)
    rgbd_heatmapstacked = np.dstack((rgb, depth_heatmap))  # Horizontal stack 
    print("RGBD Shape (heatmap-stacked): ", rgbd_heatmapstacked.shape)
    
    # RGB Segmentation
    mask = cv2.imread(mask_path)
    assert rgb.shape == mask.shape
    
    rgb_seg = rgb * mask
    print("RGB Segmentation Shape: ", rgb_seg.shape)

    rgb_seg = cv2.cvtColor(rgb_seg, cv2.COLOR_RGB2BGR)  # Convert BACK to BGR before saving
    cv2.imwrite(os.path.join(ex_root, ex_dir, rgb_fname.replace("rgb", "rgbseg")), rgb_seg)
