import os
import glob
import xml.etree.ElementTree as ET

def auto_update_xml(xml_path, parts_dir, output_xml_path=None):
    if output_xml_path is None:
        output_xml_path = xml_path

    # List up and sort split obj files
    part_files = sorted(glob.glob(os.path.join(parts_dir, "*_part_*.obj")))
    print(f"Total of {len(part_files)} parts found.")

    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    body = root.find(".//body[@name='object']")

    # 1. Add inertial tag
    inertial = body.find("inertial")
    if inertial is None:
        inertial = ET.Element("inertial", {
            "pos": "0 0 0",
            "mass": "0.1",
            "diaginertia": "0.0001 0.0001 0.0001"
        })
        body.insert(0, inertial)

    # 2. Extract color/material attributes from the visual geom (key to resolving the gray occlusion issue)
    visual_geom = body.find("geom[@name='visual']")
    mat_attr = visual_geom.get("material") if visual_geom is not None else None
    rgba_attr = visual_geom.get("rgba") if visual_geom is not None else None

    # 3. Delete old collision geom except visual geom
    for geom in list(body.findall("geom")):
        if geom.get("name") != "visual":
            body.remove(geom)

    # 4. Register Mesh and Geom
    for i, part_file in enumerate(part_files):
        filename = os.path.basename(part_file)
        mesh_name = f"obj1_p{i}"

        existing_mesh = asset.find(f"mesh[@name='{mesh_name}']")
        if existing_mesh is not None:
            asset.remove(existing_mesh)

        # Add <mesh> to <asset>
        ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "file": filename,
            "scale": "0.001 0.001 0.001"
        })

        # Configure Geom attributes (group="0" is required)
        geom_kwargs = {
            "name": f"col_{i}",
            "mesh": mesh_name,
            "type": "mesh",
            "group": "0",             # Must be set to 0 for Robosuite to recognize physical collisions
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.95 0.3 0.1",
            "solimp": "0.9 0.95 0.001",
            "solref": "0.02 1",
            "condim": "4",
        }

        # Apply the same color/material as the visual geom to the collision geom
        if mat_attr:
            geom_kwargs["material"] = mat_attr
        elif rgba_attr:
            geom_kwargs["rgba"] = rgba_attr

        ET.SubElement(body, "geom", geom_kwargs)

    # 5. Save XML
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Successfully updated {output_xml_path} file!")

if __name__ == "__main__":
    tasks = ["assembly1", "assembly2", "assembly3"]
    for task in tasks:
        asset_dir = f"../models/assets/custom_objects/fmb/meshes/{task}/"
        obj_num = 5
        for n in range(1, obj_num+1):
            input_xml_dir = os.path.join(asset_dir, f"obj{n}.xml")
            parts_dir = os.path.join(asset_dir, f"obj{n}")
            output_xml_path = os.path.join(parts_dir, f"obj{n}.xml")

            auto_update_xml(
                xml_path=input_xml_dir,
                parts_dir=parts_dir,
                output_xml_path=output_xml_path
            )