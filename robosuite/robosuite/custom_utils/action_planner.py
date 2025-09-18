import numpy as np

from robosuite.custom_utils.assembly_spatial_graph import AssemblySpatialGraph
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors


class ActionPlanner:
    """
    Mid-level action planner
    """
    def __init__(self, use_primitives, table_center):
        self.use_prim = use_primitives

        self.block_size = np.array([0.04, 0.04, 0.04])  # assuming cubic block
        self.goal_area_offset = np.array([0, 0.28, 0.025])    # looking from sideview
        self.base_pos = table_center + self.goal_area_offset
    
    def __call__(self, blueprint):
        # Get assembly structure and extract assembly order and new block positions
        print(f"{bcolors.OKCYAN}[action_planner.py | {get_clock_time()}] Computing new positions of blocks{bcolors.RESET}")

        if self.use_prim is False:    # When applying center distances instead of spatial relation primitives
            assembly_structure = blueprint["structure_relations"]
            assembly_order, new_block_positions = self._compute_positions_from_distances(assembly_structure)

        else:    # When applying spatial relation primitives
            assembly_structure = blueprint["assembly_structure"]

            # Get assembly order
            g = AssemblySpatialGraph()
            spatial_dag, assembly_order = g(assembly_structure)

            # Calculate 3d coordinates of individual blocks and log them
            new_block_positions = self._compute_positions_from_instructions(spatial_dag, assembly_order)

        print("Assembly order: ", assembly_order)
        print("New block positions: ", new_block_positions)

        return assembly_order, new_block_positions
    
    def _compute_positions_from_distances(self, dist_relationships):
        block_positions = {}
        base_blocks = []
        
        for relation in dist_relationships:
            src_block = relation["from"]
            tgt_block = relation["to"]
            center_distance = relation["center_distance"]
            relative_dist = np.array([center_distance["x"], center_distance["y"], center_distance["z"]])
            
            if src_block not in block_positions.keys():
                if len(base_blocks) == 0:
                    new_pos = self.base_pos
                else:
                    new_pos = base_blocks[-1][1] - np.array([self.block_size[0] * 1.5, 0.0, 0.0])
                base_blocks.append(tuple([src_block, new_pos]))
                block_positions[src_block] = new_pos
            
            base_position = block_positions[src_block]
            block_positions[tgt_block] = base_position + relative_dist
        
        assembly_order = [*(block_positions.keys())]

        return assembly_order, block_positions
    
    def _compute_positions_from_instructions(self, spatial_graph, assembly_order):
        """
        compute the 3d position of each block depending on its supporter
        """
        block_positions = {}
        base_blocks = []
        
        for i, block in enumerate(assembly_order):
            relationships = spatial_graph.get(block, [])
            
            for relationship in relationships:
                relation = relationship[0]
                supporters = relationship[1:]
                
                # Move blocks according to the relation to its supporter
                if relation == "base":
                    if len(base_blocks) == 0:
                        new_pos = self.base_pos
                    else:
                        # new_pos = base_blocks[-1][1] + np.array([block_size[0], 0.0, 0.0])
                        new_pos = base_blocks[-1][1] - np.array([self.block_size[0] * 1.5, 0.0, 0.0])
                    base_blocks.append(tuple([block, new_pos]))
                
                elif relation == "right":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([self.block_size[0], 0.0, 0.0])
                    
                elif relation == "left":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([-self.block_size[0], 0.0, 0.0])
                    
                elif relation == "front":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, self.block_size[1], 0.0])
                    
                elif relation == "behind":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, -self.block_size[1], 0.0])
                    
                elif relation == "top" and len(supporters) == 1:
                    # Place directly on top
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, 0.0, self.block_size[2]])
                    
                elif len(supporters) > 1:
                    assert relation == "top"
                    # Center between two supporters
                    supp1, supp2 = block_positions[supporters[0]], block_positions[supporters[1]]
                    avg_pos = (supp1 + supp2) / 2
                    avg_pos[2] += self.block_size[2]
                    new_pos = avg_pos
                
                block_positions[block] = new_pos
        
        return block_positions