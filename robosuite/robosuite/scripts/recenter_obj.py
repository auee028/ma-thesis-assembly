#!/usr/bin/env python3

from pathlib import Path
import numpy as np


def recenter_obj(input_path, output_path=None):
    """
    Recenter an OBJ mesh so that its bounding-box center becomes (0,0,0).
    """

    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(input_path.stem)
    else:
        output_path = Path(output_path)

    lines = input_path.read_text().splitlines()

    vertices = []
    vertex_indices = []

    # Read vertices
    for i, line in enumerate(lines):
        if line.startswith("v "):
            parts = line.split()
            xyz = np.array(list(map(float, parts[1:4])))
            vertices.append(xyz)
            vertex_indices.append(i)

    vertices = np.array(vertices)

    # Bounding box
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0

    print(f"{input_path.name}")
    print(f"Bounding box min : {vmin}")
    print(f"Bounding box max : {vmax}")
    print(f"Center           : {center}")

    centered = vertices - center

    # Replace vertex lines
    for idx, v in zip(vertex_indices, centered):
        lines[idx] = f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"

    output_path.write_text("\n".join(lines))

    print(f"Saved to {output_path}\n")


if __name__ == "__main__":

    # data = "assembly1"
    l_data = ["assembly2", "assembly3", "fixture", "peg", "peg_board", "peg_fixture"]
    for data in l_data:
        obj_dir = Path(f"../models/assets/custom_objects/fmb/meshes/{data}_old/") # Path(".")
        out_dir = Path(f"../models/assets/custom_objects/fmb/meshes/{data}/")

        for obj_file in sorted(obj_dir.glob("*.obj")):
            recenter_obj(obj_file, out_dir / obj_file.name)
