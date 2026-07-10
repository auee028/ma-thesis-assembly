import os
import re
import xml.etree.ElementTree as ET

FILE_ROOT = "/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb"
OBJ_FILE = "meshes/board1.obj"
MTL_FILE = "meshes/board1.mtl"
XML_FILE = "board1.xml"

# ----------------------------------------------------
# Parse MTL
# ----------------------------------------------------
materials = {}

current = None

with open(os.path.join(FILE_ROOT, MTL_FILE)) as f:
    for line in f:
        line = line.strip()

        if line.startswith("newmtl"):
            current = line.split()[1]
            materials[current] = {}

        elif line.startswith("Kd"):
            _, r, g, b = line.split()
            materials[current]["rgba"] = f"{r} {g} {b} 1"


# ----------------------------------------------------
# Parse OBJ
# ----------------------------------------------------
objects = []

current_obj = None

with open(os.path.join(FILE_ROOT, OBJ_FILE)) as f:
    for line in f:
        line = line.strip()

        if line.startswith("o "):
            current_obj = {
                "name": line.split()[1],
                "material": None,
            }
            objects.append(current_obj)

        elif line.startswith("usemtl"):
            current_obj["material"] = line.split()[1]


# ----------------------------------------------------
# Build MJCF
# ----------------------------------------------------
mujoco = ET.Element("mujoco", model="board1")

asset = ET.SubElement(mujoco, "asset")

# one mesh per OBJ object
for obj in objects:
    ET.SubElement(
        asset,
        "mesh",
        name=obj["name"],
        file=OBJ_FILE.replace('.obj', f'_{obj["name"]}.obj'),
        scale="0.001 0.001 0.001",
    )

# materials
for mat_name, vals in materials.items():
    ET.SubElement(
        asset,
        "material",
        name=mat_name,
        rgba=vals["rgba"],
    )

worldbody = ET.SubElement(mujoco, "worldbody")
body = ET.SubElement(worldbody, "body")
object_body = ET.SubElement(body, "body", name="object")

for obj in objects:

    ET.SubElement(
        object_body,
        "geom",
        type="mesh",
        mesh=obj["name"],
        material=obj["material"],
        group="1",
        contype="0",
        conaffinity="0",
    )

    ET.SubElement(
        object_body,
        "geom",
        type="mesh",
        mesh=obj["name"],
        material=obj["material"],
        group="0",
        density="500",
        friction="1 0.005 0.0001",
    )

ET.indent(mujoco)
ET.ElementTree(mujoco).write(XML_FILE)
