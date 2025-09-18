import os, sys
from PIL import Image

size = 512, 512    # width, height

if sys.argv[1:]:
    img_path = sys.argv[1:]
else:
    img_path = "/home/juhee/thesis/data/chatgpt_sandwich_ingredients.png"

# outfile = os.path.splitext(os.path.basename(img_path))[0] + f"{size[0]}x{size[1]}" + ".png"
outfile = img_path.replace(".png", f"_{size[0]}x{size[1]}.png")

try:
    im = Image.open(img_path)
    im.thumbnail(size, Image.Resampling.LANCZOS)
    im.save(outfile, "PNG")
    print("Saved Image in ", img_path)
except IOError:
    print("cannot create thumbnail for ", img_path)
