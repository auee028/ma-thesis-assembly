import numpy as np
from robosuite.custom_utils.assembly_controller import WholeBodyIKController
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors


class AssemblyPlanner:
    def __init__(self, env, block_matches, default_ee_pose=None):
        self.env = env
        self.block_matches = block_matches
        
        self.controller = WholeBodyIKController(env, default_ee_pose=default_ee_pose)
       
    def __call__(self, spatial_graph, assembly_order):
        print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Computing new positions of blocks{bcolors.RESET}")
        new_block_positions = self._compute_block_positions(spatial_graph, assembly_order)
        print("New block positions: ", new_block_positions)
        
        print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Passing the action plans to the controller{bcolors.RESET}")
        for curr_block in assembly_order:
            print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Current block: {curr_block}{bcolors.RESET}")
            body_name = self.block_matches[curr_block] + '_main'
            target_pos = new_block_positions[curr_block]
            self.controller.move_block(body_name, target_pos, target_ori=None)
        print(f"{bcolors.OKCYAN}[assembly_controller.py | {get_clock_time()}] Retracting to the robot's initial position{bcolors.RESET}")
        self.controller.retrace()
    
    def _compute_block_positions(self, spatial_graph, assembly_order):
        """
        compute the 3d position of each block depending on its supporter
        """
        block_size = np.array([0.04, 0.04, 0.04])  # assuming cubic block
        # block_size = 0.04  # assuming uniform block length
        
        table_center = self.env.sim.data.get_body_xpos("table")
        goal_area_offset = np.array([0, 0.28, 0.025])    # looking from sideview
        base_pos = table_center + goal_area_offset
        
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
                        new_pos = base_pos
                    else:
                        # new_pos = base_blocks[-1][1] + np.array([block_size[0], 0.0, 0.0])
                        new_pos = base_blocks[-1][1] - np.array([block_size[0] * 1.5, 0.0, 0.0])
                    base_blocks.append(tuple([block, new_pos]))
                
                elif relation == "right":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([block_size[0], 0.0, 0.0])
                    
                elif relation == "left":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([-block_size[0], 0.0, 0.0])
                    
                elif relation == "front":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, block_size[1], 0.0])
                    
                elif relation == "behind":
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, -block_size[1], 0.0])
                    
                elif relation == "top" and len(supporters) == 1:
                    # Place directly on top
                    supp_pos = block_positions[supporters[0]]
                    new_pos = supp_pos + np.array([0.0, 0.0, block_size[2]])
                    
                elif len(supporters) > 1:
                    assert relation == "top"
                    # Center between two supporters
                    supp1, supp2 = block_positions[supporters[0]], block_positions[supporters[1]]
                    avg_pos = (supp1 + supp2) / 2
                    avg_pos[2] += block_size[2]
                    new_pos = avg_pos
                
                block_positions[block] = new_pos
        
        return block_positions

