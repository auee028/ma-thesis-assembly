import re
from pathlib import Path

obj_file = Path("/home/juhui/ma-thesis-assembly/robosuite/robosuite/models/assets/custom_objects/fmb/meshes/board1.obj")

text = obj_file.read_text()

def increment_mat(match):
    num = int(match.group(1))
    return f"usemtl mat{num + 1}"

text = re.sub(
    r"usemtl\s+mat(\d+)",
    increment_mat,
    text,
)

obj_file.write_text(text)

print("Done!")
