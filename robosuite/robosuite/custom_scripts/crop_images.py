import os
import cv2

img_root = "/home/juhee/thesis/robosuite/robosuite/demos/output_demo_passive_viewer"
img_fname = "after_assembling_sideview_frame.png"
img_dirs = [
    "captured_images_20250703-220122_thesis_exp_qualitative_House",
    "captured_images_20250703-223357_thesis_exp_qualitative_House_tall",
    "captured_images_20250703-225056_thesis_exp_qualitative_BigTower",
    "captured_images_20250703-234127_thesis_exp_qualitative_TwinTowers",
    "captured_images_20250704-003716_thesis_exp_qualitative_BigHouse",
    "captured_images_20250704-011222_thesis_exp_qualitative_BigHousePillars",
    "captured_images_20250704-013009_thesis_exp_qualitative_BigHousePillars_vertical",
    "captured_images_20250704-021611_thesis_exp_qualitative_BigPyramid",
]

orig_size = 256
crop_size = 220
offset = 20

for d in img_dirs:
    img_path = os.path.join(img_root, d, img_fname)
    
    image = cv2.imread(img_path)
    h, w, c = image.shape
    assert h == w == orig_size
    
    cropped_image = image[h - crop_size + offset:, offset:crop_size, :]
    
    new_path = img_path.replace('.png', '_crop.png')
    print("Save image: ", new_path)
    cv2.imwrite(new_path, cropped_image)
