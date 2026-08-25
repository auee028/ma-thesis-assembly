import os
import trimesh
import argparse


def create_simple_demo1(save_path):
    """ AssemblySimpleDemo1: TWO U-SHAPED BLOCKS """
    # 1. Base board matching original Z range [-25.0, 25.0]
    base = trimesh.creation.box(extents=[230, 230, 50])

    # 2. Left and Right hole cutouts matching original Z depth [15.0, 25.0]
    # Height = 10.0, centered at Z = 20.0 (spans Z=15 to Z=25)
    # Note: Adding a tiny +1.0 height padding on top ensures exact top-surface boolean cuts
    left_hole = trimesh.creation.box(extents=[34, 104, 46])
    left_hole.apply_translation([-68, 0, 20.5])

    right_hole = trimesh.creation.box(extents=[34, 104, 46])
    right_hole.apply_translation([68, 0, 20.5])

    # 3. Perform CSG difference with manifold engine
    board = base.difference([left_hole, right_hole], engine="manifold")

    # 4. Export updated OBJ
    board.export(save_path)


def create_simple_demo2(save_path):
    """ AssemblySimpleDemo2: LONG CUBOID AND STEPPED BLOCK """
    # 1. Base solid board matching original dimensions
    base = trimesh.creation.box(extents=[230, 230, 50])

    # 2. Long Cuboid Shallow Hole (Full Length: 204, Width: 41, Depth: 10 units, Z: 40..50)
    long_hole = trimesh.creation.box(extents=[204, 41, 11])
    long_hole.apply_translation([0, 0, 20.5])

    # 3. Stepped Block Top Peg Hole (Width: 34, Length: 62, Depth: 45 units, Z: 5..50)
    # Y spans 135.5 to 197.5 (Center Y = 166.5)
    stepped_peg_1 = trimesh.creation.box(extents=[34, 62, 46])
    stepped_peg_1.apply_translation([0, 51.5, 28.0])

    # 4. Stepped Block Bottom Peg Hole (Width: 34, Length: 62, Depth: 45 units, Z: 5..50)
    # Y spans 32.5 to 94.5 (Center Y = 63.5)
    stepped_peg_2 = trimesh.creation.box(extents=[34, 62, 46])
    stepped_peg_2.apply_translation([0, -52.5, 28.0])

    # 5. CSG difference keeping long cuboid floor continuous through the center intersection
    board = base.difference(
        [long_hole, stepped_peg_1, stepped_peg_2], engine="manifold"
    )

    # 6. Export clean OBJ
    board.export(save_path)


def create_simple_demo3(save_path):
    """ AssemblySimpleDemo2: U-SHAPED BLOCK AND LONG CUBOID """
    # 1. Base solid board matching original dimensions
    base = trimesh.creation.box(extents=[230, 230, 50])

    # 2. Left U-shaped cutout, horizontally centered at X = 115 (Width: 34, Length: 104)
    u_shaped_center = trimesh.creation.box(extents=[34, 104, 46])
    u_shaped_center.apply_translation([0, 0, 20.5])

    # 3. Vertical central slot from obj1 (Width: 204, Length: 41)
    center_v_hole = trimesh.creation.box(extents=[204, 41, 11])
    center_v_hole.apply_translation([0, 0, 20.5])

    # 4. CSG difference carving out the exact obj1 hole geometry
    board = base.difference(
        [u_shaped_center, center_v_hole], engine="manifold"
    )

    # 5. Export clean OBJ
    board.export(save_path)

def create_simple_demo4(save_path):
    # 1. Base solid board matching obj1 dimensions
    base = trimesh.creation.box(extents=[230, 230, 50])

    # 2. Two Long Cuboid Slot Cutouts (Shallow Depth: 10 units, Z: 40..50)
    # Left long hole (X: -171.5 to -131.5)
    long_hole_left = trimesh.creation.box(extents=[40, 203, 11])
    long_hole_left.apply_translation([-36.5, 0, 20.5])

    # Right long hole (X: -98.5 to -58.5)
    long_hole_right = trimesh.creation.box(extents=[40, 203, 11])
    long_hole_right.apply_translation([36.5, 0, 20.5])

    # 3. Three Deeper Peg Holes for the Single Centered Trident (Deep Depth: 25 units, Z: 25..50)
    # Centered at Y = -115, Y-span = 33 (Y: -131.5 to -98.5)
    # Peg 1 (Left): X: -194.5 to -171.5 (Width: 23)
    trident_peg_1 = trimesh.creation.box(extents=[23, 33, 26])
    trident_peg_1.apply_translation([-68.0, 0, 13.0])

    # Peg 2 (Middle): X: -131.5 to -98.5 (Width: 33)
    trident_peg_2 = trimesh.creation.box(extents=[33, 33, 26])
    trident_peg_2.apply_translation([0 , 0, 13.0])

    # Peg 3 (Right): X: -58.5 to -35.5 (Width: 23)
    trident_peg_3 = trimesh.creation.box(extents=[23, 33, 26])
    trident_peg_3.apply_translation([68.0, 0, 13.0])

    # 4. Perform CSG difference using manifold engine
    cutouts = [
        long_hole_left,
        long_hole_right,
        trident_peg_1,
        trident_peg_2,
        trident_peg_3,
    ]
    board = base.difference(cutouts, engine="manifold")

    # 5. Export updated OBJ
    board.export(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_simple_demo", type=int, default=1)
    args = parser.parse_args()

    which_simple_demo = args.which_simple_demo
    
    save_dir = "../tmp"
    save_path = os.path.join(save_dir, f"obj1_simple{which_simple_demo}.obj")
    save_dir = f"../models/assets/custom_objects/fmb/meshes/simple_demo{which_simple_demo}/obj1"
    save_path = os.path.join(save_dir, "obj1.obj")

    if which_simple_demo == 1:
        create_simple_demo1(save_path)
        print(f"Saved: {save_path}")

    elif which_simple_demo == 2:
        create_simple_demo2(save_path)
        print(f"Saved: {save_path}")

    elif which_simple_demo == 3:
        create_simple_demo3(save_path)
        print(f"Saved: {save_path}")

    elif which_simple_demo == 4:
        create_simple_demo4(save_path)
        print(f"Saved: {save_path}")
        
    else:
        print("Error: No matching index.")