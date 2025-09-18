import numpy as np
from robosuite.utils.transform_utils import mat2quat, convert_quat, quat2axisangle
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors


class WholeBodyIKController:
    def __init__(self, env, default_ee_pose=None):
        self.env = env
        
        self.last_ee_pos = env.sim.data.get_site_xpos('gripper0_right_grip_site')  # same with obs['robot0_eef_pos'], but in w,x,y,z format
        self.last_ee_quat = env.sim.data.body_xquat[
            env.sim.model.body_name2id('robot0_right_hand')
        ]  # same with obs['robot0_eef_quat'], but in w,x,y,z format

        if default_ee_pose is not None:
            self.default_ee_pos = default_ee_pose[:3]
            self.default_ee_quat = default_ee_pose[3:]  # w,x,y,z format
        else:
            self.default_ee_pos = [-0.25047271, 0.01067378, 0.96573239]
            self.default_ee_quat = [0.02461963, -0.99853702, -0.01185027, 0.04666118]  # w,x,y,z format
        
    def move_to_pose(self, position, quaternion=None, gripper_action=None, num_steps=50):
        """
        Move to target pose using WholeBodyIK
        
        Args:
            position: [x,y,z]
            quaternion: [w,x,y,z]
            gripper_action: None (no change), 'open' (-1.0), or 'close' (1.0)
            num_steps: integer for the number of timesteps
        """
        if quaternion is None:
            quaternion = self.last_ee_quat
            
        # Create action vector
        action = np.zeros(7)  # 7DOF for IK_POSE (pos+ori+gripper)
        
        # Position control
        action[:3] = position
        
        # Orientation control (Axis Angle)
        quat_xyzw = convert_quat(np.array(quaternion))  # wxyz -> xyzw
        action[3:6] = quat2axisangle(quat_xyzw)
        
        # Gripper control
        if gripper_action == 'open':
            action[6] = -1
        elif gripper_action == 'close':
            action[6] = 1
        
        # Execute action
        for _ in range(num_steps):
            obs, reward, done, info = self.env.step(action)
            
            self.last_ee_pos = obs['robot0_eef_pos']
            self.last_ee_quat = convert_quat(obs['robot0_eef_quat'], to='wxyz')
        
        return done
    
    def move_block(self, block_name, target_pos, target_ori=None):
        """
        Complete a block movement sequence using WholeBodyIK
        
        Args:
            block_name: block name to move
            target_pos: [x,y,z]
            target_ori: [w,x,y,z]
        """
        # Get block position
        block_pos = self.env.sim.data.get_body_xpos(block_name)
        block_quat = self.env.sim.data.get_body_xquat(block_name)
        
        # Define waypoints
        z_offset = 0.05    # height as block size
        above_block = block_pos.copy()
        above_block[2] += z_offset
        
        above_target = target_pos.copy()
        above_target[2] += z_offset
        
        # Execute movement sequence
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Moving to approach position{bcolors.RESET}")
        # done = self.move_to_pose(above_block, quaternion=block_quat)
        done = self.move_to_pose(above_block)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Descending to block{bcolors.RESET}")
        # done = self.move_to_pose(block_pos, quaternion=block_quat, num_steps=20)
        done = self.move_to_pose(block_pos, num_steps=30)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Closing gripper{bcolors.RESET}")
        # done = self.move_to_pose(block_pos, quaternion=block_quat, gripper_action='close', num_steps=10)
        done = self.move_to_pose(block_pos, gripper_action='close', num_steps=10)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Lifting block{bcolors.RESET}")
        # done = self.move_to_pose(above_block, quaternion=block_quat, num_steps=20)
        done = self.move_to_pose(above_block, num_steps=30)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Moving to target area{bcolors.RESET}")
        above_target[2] += z_offset
        done = self.move_to_pose(above_target, quaternion=target_ori)
        if done: return
        
        # print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Placing block{bcolors.RESET}")
        # done = self.move_to_pose(target_pos, quaternion=target_ori, num_steps=20)
        # if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Descending to target{bcolors.RESET}")
        above_target[2] -= z_offset
        # done = self.move_to_pose(above_target, quaternion=block_quat, num_steps=20)
        done = self.move_to_pose(above_target, num_steps=30)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Opening gripper{bcolors.RESET}")
        done = self.move_to_pose(above_target, gripper_action='open', num_steps=10)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Ascending from target{bcolors.RESET}")
        above_target[2] += 2 * z_offset
        # done = self.move_to_pose(target_pos, quaternion=block_quat, num_steps=20)
        done = self.move_to_pose(above_target, num_steps=20)
        if done: return
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Retracting{bcolors.RESET}")
        done = self.move_to_pose(self.default_ee_pos, num_steps=40)
        if done: return
