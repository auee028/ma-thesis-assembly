import numpy as np
from robosuite.custom_utils.assembly_controller import WholeBodyIKController
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors

FMB_TASKS = ["assembly1", "assembly2", "assembly3"]


class AssemblyPlanner:
    def __init__(self, env, block_matches, default_ee_pose=None):
        self.env = env
        self.env_name = env.__class__.__name__
        self.block_matches = block_matches
        
        self.controller = WholeBodyIKController(env, default_ee_pose=default_ee_pose)

        # self.fixture_offsets = {
        #     # x-axis
        #     "left": np.array([-0.08, 0.0, 0.0]),
        #     "right": np.array([0.08, 0.0, 0.0]),

        #     # center
        #     "center": np.array([0.0, 0.0, 0.0]),

        #     # y-axis
        #     "front": np.array([0.0, 0.08, 0.0]),
        #     "back": np.array([0.0, -0.08, 0.0]),

        #     # optional diagonal
        #     "front-left": np.array([-0.08, 0.08, 0.0]),
        #     "front-right": np.array([0.08, 0.08, 0.0]),
        #     "back-left": np.array([-0.08, -0.08, 0.0]),
        #     "back-right": np.array([0.08, -0.08, 0.0]),
        # }

        self.hole_sites = {}
        for i in range(env.sim.model.nsite):
            # print(i, env.sim.model.site_id2name(i))
            site_name = env.sim.model.site_id2name(i)
            site_id = env.sim.model.site_name2id(site_name)
            site_pos = env.sim.data.site_xpos[site_id]
            # print(site_name, site_pos)
            if 'obj1_hole' in site_name:    # board frame
                self.hole_sites[site_name] = site_pos
       
    def __call__(self, spatial_graph, assembly_order):
        print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Computing new positions of blocks{bcolors.RESET}")
        new_block_positions = self._compute_block_positions(spatial_graph, assembly_order)
        print("New block positions: ", new_block_positions)
        
        print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Passing the action plans to the controller{bcolors.RESET}")
        if self.env_name.lower() in FMB_TASKS:
            assembly_order = assembly_order[1:]   # skip the first obj (fixture) for the FMB tasks
        for curr_block in assembly_order:
            print(f"{bcolors.OKCYAN}[assembly_planner.py | {get_clock_time()}] Current block: {curr_block}{bcolors.RESET}")
            body_name = self.block_matches[curr_block] + '_main'
            target_pos = new_block_positions[curr_block]
            self.controller.move_block(body_name, target_pos, target_ori=None)
        print(f"{bcolors.OKCYAN}[assembly_controller.py | {get_clock_time()}] Retracting to the robot's initial position{bcolors.RESET}")
        self.controller.retrace()
    
    def _compute_block_positions(self, spatial_graph, assembly_order):
        """
        compute the 3d position of each block depending on its supporter and sites
        """
        block_size = np.array([0.04, 0.04, 0.04])  # assuming cubic block
        table_center = self.env.sim.data.get_body_xpos("table")

        # Initialize base_pos and goal_area_offset
        goal_area_offset = np.array([0, 0.28, 0.025])    # looking from sideview
        base_pos = table_center + goal_area_offset
        
        if self.env_name.lower() in FMB_TASKS:
            frame_block = assembly_order[0]
            body_name = self.block_matches[frame_block] + "_main"
            base_pos = self.env.sim.data.get_body_xpos(body_name)
        
        block_positions = {}
        base_blocks = []
        
        for i, block in enumerate(assembly_order):
            relationships = spatial_graph.get(block, [])
            
            # Extract relationships (e.g. ['left', 'center'] form)
            rel_types = [rel[0] for rel in relationships]
            supporters = [rel[1] for rel in relationships if len(rel) > 1]
            first_supp = supporters[0] if supporters else None

            if self.env_name.lower() not in FMB_TASKS:
                # Non-FMB tasks
                for relationship in relationships:
                    relation = relationship[0]
                    supps = relationship[1:]

                    if relation == "base":
                        if len(base_blocks) == 0:
                            new_pos = base_pos
                        else:
                            new_pos = base_blocks[-1][1] - np.array([block_size[0] * 1.5, 0.0, 0.0])
                        base_blocks.append(tuple([block, new_pos]))
                    elif relation == "right":
                        supp_pos = block_positions[supps[0]]
                        new_pos = supp_pos + np.array([block_size[0], 0.0, 0.0])
                    elif relation == "left":
                        supp_pos = block_positions[supps[0]]
                        new_pos = supp_pos + np.array([-block_size[0], 0.0, 0.0])
                    elif relation == "front":
                        supp_pos = block_positions[supps[0]]
                        new_pos = supp_pos + np.array([0.0, block_size[1], 0.0])
                    elif relation == "behind":
                        supp_pos = block_positions[supps[0]]
                        new_pos = supp_pos + np.array([0.0, -block_size[1], 0.0])
                    elif relation == "top" and len(supps) == 1:
                        supp_pos = block_positions[supps[0]]
                        new_pos = supp_pos + np.array([0.0, 0.0, block_size[2]])
                    elif len(supps) > 1:
                        assert relation == "top"
                        supp1, supp2 = block_positions[supps[0]], block_positions[supps[1]]
                        avg_pos = (supp1 + supp2) / 2
                        avg_pos[2] += block_size[2]
                        new_pos = avg_pos

            else:   # FMB tasks
                # 1. Base (Frame/Fixture) pos assignment
                if "base" in rel_types:
                    new_pos = base_pos
                    base_blocks.append((block, new_pos))

                # 2. Assigned in a site hole of Frame(block0)
                elif first_supp == "block0":
                    # left hole: ('left', 'center') multiple relations or 'left' included
                    if "left" in rel_types:
                        keyword = "left"
                        # Returns the list value, or None if no key contains the keyword
                        hole_site_pos = next((val for key, val in self.hole_sites.items() if keyword in key), None)
                        new_pos = hole_site_pos + np.array([0.0, 0.0, block_size[2] / 2])
                    # right hole: ('right', 'center') multiple relations or 'right' included
                    elif "right" in rel_types:
                        keyword = "right"
                        hole_site_pos = next((val for key, val in self.hole_sites.items() if keyword in key), None)
                        new_pos = hole_site_pos + np.array([0.0, 0.0, block_size[2] / 2])
                    # front hole: ('front', 'center') multiple relations or 'front' included
                    elif "front" in rel_types:
                        keyword = "front"
                        hole_site_pos = next((val for key, val in self.hole_sites.items() if keyword in key), None)
                        new_pos = hole_site_pos + np.array([0.0, 0.0, block_size[2] / 2])
                    # behind hole: ('behind', 'center') multiple relations or 'behind' included
                    elif "behind" in rel_types:
                        keyword = "behind"
                        hole_site_pos = next((val for key, val in self.hole_sites.items() if keyword in key), None)
                        new_pos = hole_site_pos + np.array([0.0, 0.0, block_size[2] / 2])
                    # center hole: only 'center'
                    elif "center" in rel_types:
                        keyword = "center"
                        hole_site_pos = next((val for key, val in self.hole_sites.items() if keyword in key), None)
                        new_pos = hole_site_pos + np.array([0.0, 0.0, block_size[2] / 2])
                    else:
                        new_pos = base_pos

                # 3. Assigned based on other inserted blocks (e.g. block4 -> right of block3)
                else:
                    supp_pos = block_positions[first_supp]
                    if "right" in rel_types:
                        new_pos = supp_pos + np.array([block_size[0], 0.0, 0.0])
                    elif "left" in rel_types:
                        new_pos = supp_pos + np.array([-block_size[0], 0.0, 0.0])
                    elif "front" in rel_types:
                        new_pos = supp_pos + np.array([0.0, block_size[1], 0.0])
                    elif "behind" in rel_types:
                        new_pos = supp_pos + np.array([0.0, -block_size[1], 0.0])
                    # elif "above" in rel_types:
                    #     new_pos = block_positions
                    else:
                        new_pos = supp_pos + np.array([0.0, 0.0, block_size[2]])

            block_positions[block] = new_pos
        
        return block_positions

