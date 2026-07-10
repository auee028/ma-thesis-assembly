import os
import json
from pathlib import Path

OBJ_FILE = "/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/meshes/board1.obj"
OUT_DIR = "/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/meshes/board1"

OBJ_FILE = Path(OBJ_FILE)
OUT_DIR = Path(OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

vertices = []

objects = []

current = None

with open(OBJ_FILE) as f:
    for line in f:
        line = line.rstrip()

        if line.startswith("v "):
            vertices.append(line)

        elif line.startswith("o "):

            if current is not None:
                objects.append(current)

            current = {
                "name": line.split()[1],
                "material": None,
                "faces": [],
                "vertex_ids": set(),
            }

        elif line.startswith("usemtl"):
            current["material"] = line.split()[1]

        elif line.startswith("f "):

            ids = []

            for token in line.split()[1:]:

                # works for:
                #
                # f 1 2 3
                # f 1//2 2//3 3//4
                # f 1/2/3

                idx = int(token.split("/")[0])

                ids.append(idx)

                current["vertex_ids"].add(idx)

            current["faces"].append(ids)

if current is not None:
    objects.append(current)

material_map = {}

for obj in objects:

    used = sorted(obj["vertex_ids"])

    old2new = {
        old: i + 1
        for i, old in enumerate(used)
    }

    outfile = OUT_DIR / f"{obj['name']}.obj"

    with open(outfile, "w") as f:

        f.write(f"mtllib {OBJ_FILE.with_suffix('.mtl').name}\n")
        f.write(f"o {obj['name']}\n")

        # write vertices

        for vid in used:
            f.write(vertices[vid - 1] + "\n")

        if obj["material"]:
            f.write(f"usemtl {obj['material']}\n")

        # write faces

        for face in obj["faces"]:

            face = [str(old2new[v]) for v in face]

            f.write("f " + " ".join(face) + "\n")

    material_map[obj["name"]] = obj["material"]

with open(OUT_DIR / "materials.json", "w") as f:
    json.dump(material_map, f, indent=4)

print(f"Wrote {len(objects)} OBJ files.")
