import os
import glob
import numpy as np
import open3d as o3d

camera_name = "agentview"
pcd_dir = sorted(glob.glob("/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer/*"))[-1]
pcd_path = glob.glob(os.path.join(pcd_dir, f"{camera_name}_*_scene.ply"))[0]
print(pcd_path)

# Load point cloud
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
print(points.min(), points.max())
print(colors.min(), colors.max())
o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.5,
    front=[0.027865, -0.295117, 0.955055],
    up=[0.004633, 0.955454, 0.295105],
    lookat=np.mean(np.asarray(pcd.points), axis=0),
)

"""
# Visualize with custom settings
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add point cloud
vis.add_geometry(pcd)

# Access view control
view_control = vis.get_view_control()
# view_control.set_zoom(1.2)

# Set lookat point (use mean position for best effect)
# lookat_point = np.mean(np.asarray(pcd.points), axis=0)
lookat_point = np.asarray(pcd.points)[0]
view_control.set_lookat(lookat_point)

# Run visualization
vis.run()
vis.destroy_window()
"""
