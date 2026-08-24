import trimesh

# TWO U-SHAPED BLOCKS
# 1. Base board matching original Z range [-25.0, 25.0]
base = trimesh.creation.box(extents=[230, 230, 50])

# 2. Left and Right hole cutouts matching original Z depth [15.0, 25.0]
# Height = 10.0, centered at Z = 20.0 (spans Z=15 to Z=25)
# Note: Adding a tiny +1.0 height padding on top ensures exact top-surface boolean cuts
left_hole = trimesh.creation.box(extents=[34, 104, 11])
left_hole.apply_translation([-68, 0, 20.5])

right_hole = trimesh.creation.box(extents=[34, 104, 11])
right_hole.apply_translation([68, 0, 20.5])

# 3. Perform CSG difference with manifold engine
board = base.difference([left_hole, right_hole], engine="manifold")

# 4. Export updated OBJ
board.export("../tmp/obj1_simple1.obj")

