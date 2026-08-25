import os
import trimesh
import coacd

def decompose_and_generate_xml(obj_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    
    # 1. Load mesh
    mesh = trimesh.load(obj_path)
    
    # 2. Execute CoACD  (Convex Decomposition)
    # The lower the threshold value is, the more sophisticated it is split (default 0.05)
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # parts = coacd.run_coacd(coacd_mesh, threshold=0.05)
    parts = coacd.run_coacd(coacd_mesh, threshold=0.01)
    
    # 3. Save the split meshes in .obj files
    saved_files = []
    for i, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        part_filename = f"{base_name}_part_{i}.obj"
        part_path = os.path.join(output_dir, part_filename)
        part_mesh.export(part_path)
        saved_files.append(part_filename)
        print(f"Saved: {part_path}")

    # # 4. Print XML tags
    # print("\n" + "="*50)
    # print(" Copy the following contents inside of the XML <asset> tag:")
    # print("="*50)
    # print(f'<!-- Visual Mesh -->')
    # print(f'<mesh name="{base_name}_visual" file="{obj_path}" scale="0.001 0.001 0.001" />')
    # print(f'<!-- Collision Parts -->')
    # for i, fname in enumerate(saved_files):
    #     print(f'<mesh name="{base_name}_p{i}" file="{fname}" scale="0.001 0.001 0.001" />')

    # print("\n" + "="*50)
    # print(" Copy the following contents inside of the XML <body name=\"object\"> tag:")
    # print("="*50)
    # print(f'<geom name="visual" mesh="{base_name}_visual" type="mesh" group="1" contype="0" conaffinity="0" material="mat1" />')
    # for i in range(len(saved_files)):
    #     # print(f'<geom name="col_{i}" mesh="{base_name}_p{i}" type="mesh" group="0" density="50" friction="0.95 0.3 0.1" solimp="0.9 0.95 0.001" solref="0.02 1" condim="3" />')
    #     print(f'<geom name="col_{i}" mesh="{base_name}_p{i}" type="mesh" group="0" contype="1" conaffinity="1" friction="0.95 0.3 0.1" solimp="0.9 0.95 0.001" solref="0.02 1" condim="4" material="mat1" />')

    # 4. Run auto_update_xml.py


if __name__ == "__main__":
    # Enter the .obj file path
    # tasks = ["assembly1", "assembly2", "assembly3"]
    tasks = [f"simple_demo{i}" for i in range(1, 5)]

    for task in tasks:
        obj_dir = f"../models/assets/custom_objects/fmb/meshes/{task}"

        _, dirs, _ = next(os.walk(obj_dir))
        num_obj = len(dirs)
        for i in range(1, num_obj+1):
            out_dir = os.path.join(obj_dir, f"obj{i}")  # "decomposed parts"
            decompose_and_generate_xml(
                # obj_path=os.path.join(obj_dir, f"obj{i}.obj"),
                obj_path=os.path.join(obj_dir, f"obj{i}", f"obj{i}.obj"),
                output_dir=out_dir
            )
