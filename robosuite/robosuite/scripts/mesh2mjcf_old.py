from pathlib import Path
import xml.etree.ElementTree as ET

# ============================================================
# Change these
# ============================================================

MODEL_NAME = "fixture"      # assembly1, assembly2, assembly3, peg, peg_board, fixture

OBJ_FILE = Path(
    f"/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/meshes/{MODEL_NAME}.obj"
)

OUTPUT_DIR = Path(
    f"/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/meshes/{MODEL_NAME}"
)
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_XML_FILE = Path(
    f"/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/{MODEL_NAME}.xml"
)

SCALE = "0.001 0.001 0.001"

# ============================================================
# Parse MTL
# ============================================================

MTL_FILE = OBJ_FILE.with_suffix(".mtl")

materials = {}
current = None

with open(MTL_FILE) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        if line.startswith("newmtl"):

            current = line.split()[1]

            materials[current] = {
                "rgba": "0.8 0.8 0.8 1"
            }

        elif line.startswith("Kd"):

            _, r, g, b = line.split()

            materials[current]["rgba"] = f"{r} {g} {b} 1"

# ============================================================
# Parse OBJ
# ============================================================

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

                idx = int(token.split("/")[0])

                ids.append(idx)
                current["vertex_ids"].add(idx)

            current["faces"].append(ids)

if current is not None:
    objects.append(current)

print(f"Found {len(objects)} objects")

# ============================================================
# Write split OBJ files
# ============================================================

for obj in objects:

    used = sorted(obj["vertex_ids"])

    old2new = {
        old: i + 1
        for i, old in enumerate(used)
    }

    outfile = OUTPUT_DIR / f"{obj['name']}.obj"

    with open(outfile, "w") as f:

        f.write(f"mtllib {MTL_FILE.name}\n")
        f.write(f"o {obj['name']}\n\n")

        for vid in used:
            f.write(vertices[vid - 1] + "\n")

        f.write("\n")

        if obj["material"] is not None:
            f.write(f"usemtl {obj['material']}\n")

        for face in obj["faces"]:

            face = [str(old2new[v]) for v in face]

            f.write("f " + " ".join(face) + "\n")

print("Split OBJ files written.")

# ============================================================
# Generate MJCF
# ============================================================

mujoco = ET.Element("mujoco", model=MODEL_NAME)

asset = ET.SubElement(mujoco, "asset")

# ------------------------------------------------------------
# Mesh assets
# ------------------------------------------------------------

for obj in objects:

    ET.SubElement(
        asset,
        "mesh",
        name=obj["name"],
        file=f"meshes/{MODEL_NAME}/{obj['name']}.obj",
        scale=SCALE,
    )

# ------------------------------------------------------------
# Materials
# ------------------------------------------------------------

for mat_name, mat in materials.items():

    ET.SubElement(
        asset,
        "material",
        name=mat_name,
        rgba=mat["rgba"],
    )

# ------------------------------------------------------------
# Worldbody
# ------------------------------------------------------------

worldbody = ET.SubElement(mujoco, "worldbody")

body = ET.SubElement(worldbody, "body")

object_body = ET.SubElement(
    body,
    "body",
    name="object",
)

# ------------------------------------------------------------
# Visual geoms
# ------------------------------------------------------------

for obj in objects:

    attrib = {
        "name": obj["name"],
        "mesh": obj["name"],
        "type": "mesh",
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    }

    if obj["material"] in materials:
        attrib["material"] = obj["material"]

    ET.SubElement(
        object_body,
        "geom",
        attrib,
    )

# ------------------------------------------------------------
# Collision geoms
# ------------------------------------------------------------

for obj in objects:

    ET.SubElement(
        object_body,
        "geom",
        name=f"{obj['name']}",
        mesh=obj["name"],
        type="mesh",
        density="500",
        friction="1 0.005 0.0001",
        group="0",
    )

# ------------------------------------------------------------
# robosuite sites
# ------------------------------------------------------------

ET.SubElement(
    object_body,
    "site",
    name="bottom_site",
    pos="0 0 -0.025",
    size="0.005",
    rgba="0 0 0 0",
)

ET.SubElement(
    object_body,
    "site",
    name="top_site",
    pos="0 0 0.025",
    size="0.005",
    rgba="0 0 0 0",
)

ET.SubElement(
    object_body,
    "site",
    name="horizontal_radius_site",
    pos="0.03 0.03 0",
    size="0.005",
    rgba="0 0 0 0",
)

# ------------------------------------------------------------

ET.indent(mujoco)

ET.ElementTree(mujoco).write(
    OUT_XML_FILE,
    encoding="utf-8",
    xml_declaration=True,
)

print(f"Wrote {OUT_XML_FILE}")
