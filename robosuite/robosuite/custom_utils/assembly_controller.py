import numpy as np
import time
from robosuite.utils.transform_utils import mat2quat, convert_quat, quat2axisangle, axisangle2quat, quat_multiply, quat_distance
from robosuite.custom_utils.voxposer_utils import get_clock_time, bcolors


class WholeBodyIKController:
    def __init__(self, env, default_ee_pose=None):
        self.env = env
        
        self.last_ee_pos = self._get_gripper_pos()
        self.last_ee_quat = self._get_gripper_quat()

        if default_ee_pose is not None:
            self.default_ee_pos = default_ee_pose[:3]
            self.default_ee_quat = default_ee_pose[3:]  # w,x,y,z format
        else:
            self.default_ee_pos = [-0.2, 0.0, 0.99]
            self.default_ee_quat = [0.0, -1.0, 0.0, 0.0]  # w,x,y,z format
        
        self._action_terminated = False
    
    def _get_eef_pos(self):
        return self._get_gripper_pos()
    
    def _get_eef_quat(self):
        return self.env.sim.data.body_xquat[
            self.env.sim.model.body_name2id('robot0_right_hand')
        ]  # same with obs['robot0_eef_quat'], but in w,x,y,z format
    
    def _get_gripper_pos(self):
        return self.env.sim.data.get_site_xpos('gripper0_right_grip_site')  # same with obs['robot0_eef_pos']
    
    def _get_gripper_quat(self):
        return self.env.sim.data.body_xquat[
            self.env.sim.model.body_name2id('gripper0_right_eef')
        ]  # same with obs['robot0_eef_quat_site'], but in w,x,y,z format
    
    def _get_block_pos(self, block_name):
        return self.env.sim.data.get_body_xpos(block_name)
    
    def _get_block_quat(self, block_name, to='wxyz'):
        if to == 'xyzw':
            return convert_quat(
                self.env.sim.data.get_body_xquat(block_name),
                to='xyzw'
            )
        return self.env.sim.data.get_body_xquat(block_name)    # wxyz format
    
    def _check_action_termination(
        self,
        obs,
        goal_pos=None,
        goal_quat=None,
        phase=None,
        block_name=None,
        pos_thresh=0.006,
        ori_thresh=1.5,
        height_thresh=0.01,
        open_thresh=0.023,
        open_limit=0.03,
        num_steps=None):
        """
        Checks if a motion phase is successfully completed.
        The termination flag is set to 'True' if termination condition is met.
        
        Args:
            obs: current observation
            goal_pos: desired target position for the end-effector
            goal_quat: desired orientation (x,y,z,w) of the end-effector
            phase: string to identify phase ("reach", "grasp", "lift", "place", "retract", "orient")
            block_name: relevant object for grasp/lift/place
            pos_thresh: position tolerance
            ori_thresh: orientation tolerance (radians)
            height_thresh: min lifting height to consider successful grasp
            open_thresh: gripper opening tolerance
            open_limit: min gripper opening to consider fallen block
            num_steps: alternative way of excutution to checking termination condition
        """
        eef_pos = obs['robot0_eef_pos']    # xyzw order
        eef_quat = obs['robot0_eef_quat']    # xyzw order

        if phase == "reach":
            # Check orientation match
            if goal_quat is not None:
                delta_quat = quat2axisangle(quat_distance(eef_quat, goal_quat))
                ori_aligned = np.linalg.norm(delta_quat) < ori_thresh
            else:
                ori_aligned = True
            
            pos_reached = np.linalg.norm(eef_pos - goal_pos) < pos_thresh
            
            self._action_terminated = ori_aligned and pos_reached

        elif phase == "grasp":
            # Use the env's _check_grasp method
            gripper = self.env.robots[0].gripper
            object_geoms = block_name
            
            self._action_terminated = self.env._check_grasp(gripper, object_geoms)

        elif phase == "lift":
            block_pos = self._get_block_pos(block_name)
            
            self._action_terminated = abs(goal_pos[2] - block_pos[2]) < height_thresh

        elif phase == "place":
            block_pos = self._get_block_pos(block_name)
            dist = np.linalg.norm(eef_pos - block_pos)
            gripper_open = dist > open_thresh and dist < open_limit
            
            self._action_terminated = gripper_open

        else:
            raise ValueError(f"Unknown phase type: {phase}")    
        
    def move_to_pose(
        self, 
        position, 
        quaternion=None, 
        gripper_action=None, 
        phase=None, 
        block_name=None, 
        pos_thresh=0.006, 
        ori_thresh=0.45, 
        height_thresh=0.01, 
        open_thresh=0.023, 
        open_limit=0.03, 
        timeout=15.0, 
        num_steps=None):
        """
        Move to target pose using WholeBodyIK
        
        Args:
            position: [x,y,z]
            quaternion: [w,x,y,z]
            gripper_action: None (no change), 'open' (-1.0), or 'close' (1.0)
            phase: motion phase string for checking termination ('reach', 'grasp', 'lift', 'place', 'retract')
            block_name: if interacting with a block (needed for grasp/lift/place)
            timeout: action execution timeout
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
        # else, it is zero which means no change
        
        start_time = time.time()
        
        # Execute action
        if num_steps:    # Force execution
            for _ in range(num_steps):
                obs, reward, done, info = self.env.step(action)
                self.last_ee_pos = obs['robot0_eef_pos']
                self.last_ee_quat = convert_quat(obs['robot0_eef_quat'], to='wxyz')
        else:     # More natural execution phases expected
            while not self._action_terminated:
                obs, reward, done, info = self.env.step(action)
                
                # Check if the action termination condition is met
                self._check_action_termination(obs, goal_pos=position, goal_quat=quat_xyzw, phase=phase, block_name=block_name, pos_thresh=pos_thresh, ori_thresh=ori_thresh, height_thresh=height_thresh, open_thresh=open_thresh, open_limit=open_limit, num_steps=num_steps)
                # Timeout condition
                if time.time() - start_time > timeout:
                    print("[WARNING] Action timed out")
                    break
                
                self.last_ee_pos = obs['robot0_eef_pos']
                self.last_ee_quat = convert_quat(obs['robot0_eef_quat'], to='wxyz')
            self._action_terminated = False    # initialize the flag
    
    def move_block(self, block_name, target_pos, target_ori=None):
        """
        Complete a block movement sequence using WholeBodyIK
        
        Args:
            block_name: block name to move
            target_pos: [x,y,z]
            target_ori: [w,x,y,z]
        """
        # Get block pose
        block_pos = self._get_block_pos(block_name)
        block_quat = self._get_block_quat(block_name)
        
        # Set target orientation as default if not given
        if target_ori is None:
            target_ori = self.default_ee_quat
        
        # Define waypoints
        z_offset = 0.042
        above_block = block_pos.copy()
        above_block[2] += z_offset
        more_above_block = block_pos.copy()
        more_above_block[2] += 3.5 * z_offset
        
        above_target = target_pos.copy()
        above_target[2] += z_offset
        more_above_target = target_pos.copy()
        more_above_target[2] += 3.5 * z_offset
        
        # Execute movement sequence
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Moving to approach position{bcolors.RESET}")
        self.move_to_pose(more_above_block, phase='reach', num_steps=40)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Descending to block{bcolors.RESET}")
        block_quat_xyzw = convert_quat(block_quat, to='xyzw')
        quat_180_x = axisangle2quat([np.pi, 0, 0])   # 180° around X-axis (flip)
        quat_90_z = axisangle2quat([0, 0, -np.pi / 2])   # -90° around Z-axis
        grasp_quat = convert_quat(
            quat_multiply(quat_180_x, quat_multiply(quat_90_z, block_quat_xyzw)),
            to='wxyz'
        )
        place_quat = convert_quat(
            quat_multiply(quat_90_z, convert_quat(np.array(self.default_ee_quat))),
            to='wxyz'
        )
        self.move_to_pose(block_pos, quaternion=grasp_quat, phase='reach', ori_thresh=1.6, num_steps=35)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Closing gripper{bcolors.RESET}")
        self.move_to_pose(block_pos, quaternion=grasp_quat, gripper_action='close', phase='grasp', num_steps=10)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Lifting block{bcolors.RESET}")
        self.move_to_pose(more_above_block, quaternion=place_quat, gripper_action='close', block_name=block_name, phase='lift', num_steps=20)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Moving to target area{bcolors.RESET}")
        self.move_to_pose(more_above_target, quaternion=place_quat, phase='reach', num_steps=50)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Descending to target{bcolors.RESET}")
        self.move_to_pose(above_target, quaternion=place_quat, phase='reach', num_steps=30)
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Opening gripper{bcolors.RESET}")
        self.move_to_pose(above_target, quaternion=place_quat, gripper_action='open', block_name=block_name, phase='place', num_steps=10)
         
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Ascending from target{bcolors.RESET}")
        self.move_to_pose(more_above_target, quaternion=place_quat, phase='reach', num_steps=30)
    
    def retrace(self, target_pos=None, target_ori=None, num_steps=50):
        """
        Move the robot to its initial pose using WholeBodyIK
        
        Args:
            target_pos: [x,y,z]
            target_ori: [w,x,y,z]
            num_steps: time steps to execute retracing
        """
        # Set target orientation as default if not given
        if target_pos is None:
            target_pos = self.default_ee_pos
        
        print(f"{bcolors.OKGREEN}[assembly_controller.py | {get_clock_time()}] Retracting{bcolors.RESET}")
        self.move_to_pose(target_pos, quaternion=target_ori, phase='reach', num_steps=num_steps)
