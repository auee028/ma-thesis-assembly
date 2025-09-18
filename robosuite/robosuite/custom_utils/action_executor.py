import numpy as np
from robosuite.custom_utils.assembly_controller import WholeBodyIKController
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors


class ActionExecutor:
    def __init__(self, env, block_matches, default_ee_pose=None):
        self.env = env
        self.block_matches = block_matches
        
        self.controller = WholeBodyIKController(env, default_ee_pose=default_ee_pose)
       
    def __call__(self, assembly_order, new_block_positions):        
        print(f"{bcolors.OKCYAN}[action_executor.py | {get_clock_time()}] Passing the action plans to the controller{bcolors.RESET}")

        for curr_block in assembly_order:
            print(f"{bcolors.OKCYAN}[action_executor.py | {get_clock_time()}] Current block: {curr_block}{bcolors.RESET}")
            body_name = self.block_matches[curr_block] + '_main'
            target_pos = new_block_positions[curr_block]
            self.controller.move_block(body_name, target_pos, target_ori=None)

        # print(f"{bcolors.OKCYAN}[assembly_controller.py | {get_clock_time()}] Retracting to the robot's initial position{bcolors.RESET}")
        self.controller.retrace()