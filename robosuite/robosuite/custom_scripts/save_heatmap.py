import os
import numpy as np
import cv2


# p = "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_20250704-150146_BigHousePillars_thesis_fig_aux/assembly_input_img/assembly_depth_milli_0.png"
paths = [
    "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_20250507-155702_ex1/agentview_frame_0_depth_milli.png",
    "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_20250507-155752_ex2/agentview_frame_0_depth_milli.png",
    "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_20250507-160214_ex3/agentview_frame_0_depth_milli.png",
    "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/captured_images_20250516-002905_ex4/agentview_frame_0_depth_milli.png"
]

for p in paths:
    depth = cv2.imread(p, cv2.IMREAD_UNCHANGED) 
    print(f"Depth shape: {depth.shape}")

    # Save depth image in grayscale scaled in 0-255
    # depth_min = 560
    depth_min = 450
    depth_max = 980
    img = np.clip(depth, depth_min, depth_max)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(p.replace('milli', 'milli_clipped'), img)

    # Save depth image in a heatmap
    depth_min = 580
    img = np.clip(depth, depth_min, depth_max)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(p.replace("milli", "heatmap_clipped"), heatmap)
