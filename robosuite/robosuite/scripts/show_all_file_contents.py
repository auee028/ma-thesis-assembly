import os
import glob

file_dir = "../models/assets/custom_objects/fmb/meshes/assembly1/obj1"
save_path = os.path.join("../tmp", "print_file_contents.txt")

l = sorted(
    glob.glob(os.path.join(file_dir, "*_part_*.obj")),
    key=lambda x: int(os.path.basename(x).split("_part_")[-1].split(".obj")[0])
)

str_contents = ""
for file_path in l:
    str_contents += f"`{os.path.basename(file_path)}`:\n"
    str_contents += "```\n"
    with open(file_path, 'r') as f:
        contents = f.readlines()
        for c in contents:
            if 'http' in c: continue
            str_contents += c
    str_contents += "```\n"

with open(save_path, 'w') as f:
    f.write(str_contents)