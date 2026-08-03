from collections import OrderedDict

import numpy as np
import random

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
# from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.objects import FMBObject


class Assembly1(ManipulationEnv):
    """
    This class corresponds to the customized House task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        mujoco_passive_viewer=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # # object placement initializer
        # self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            mujoco_passive_viewer=mujoco_passive_viewer
        )
    
    def reward(self, action):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # TODO: This is an example. To do to define a good reward function
        reward = self.staged_rewards()

        return reward

    def staged_rewards(self, action=None):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            float: reward value
        """
        # TODO
        reward = 0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize the objects of FMB Assembly 1
        self.obj1 = FMBObject(
            name="obj1",	# red (frame)
            fmb_xml_file="custom_objects/fmb/meshes/assembly1/obj1.xml",
        )
        self.obj2 = FMBObject(
            name="obj2",	# purple
            fmb_xml_file="custom_objects/fmb/meshes/assembly1/obj2.xml",
        )
        self.obj3 = FMBObject(
            name="obj3",	# yellow
            fmb_xml_file="custom_objects/fmb/meshes/assembly1/obj3.xml",
        )
        self.obj4 = FMBObject(
            name="obj4",	# green
            fmb_xml_file="custom_objects/fmb/meshes/assembly1/obj4.xml",
        )
        self.obj5 = FMBObject(
            name="obj5",	# blue
            fmb_xml_file="custom_objects/fmb/meshes/assembly1/obj5.xml",
        )
        
        self.objects = [self.obj1, self.obj2, self.obj3, self.obj4, self.obj5]
        
        '''
        # # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(self.objects)
        # else:
        #     self.placement_initializer = UniformRandomSampler(
        #         name="ObjectSampler",
        #         mujoco_objects=self.objects,
        #         x_range=[-0.2, 0.2],
        #         y_range=[-0.2, 0.2],
        #         rotation_axis='z',
        #         rotation=None,
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=self.table_offset,
        #         z_offset=0.01,
        #     )
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # -----------------------------
        # Frame (obj1)
        # -----------------------------
        frame_sampler = UniformRandomSampler(
            name="FrameSampler",
            mujoco_objects=[self.obj1],
            x_range=[0.0, 0.0],
            y_range=[0.2, 0.2],        # left side
            rotation=None,
            rotation_axis="z",
            ensure_valid_placement=True,
            ensure_object_boundary_in_range=False,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        # -----------------------------
        # Parts (obj2-obj5)
        # -----------------------------
        parts_sampler = UniformRandomSampler(
            name="PartsSampler",
            mujoco_objects=[self.obj2, self.obj3, self.obj4, self.obj5],
            x_range=[-0.12, 0.12],
            y_range=[-0.12, 0.12],      # right side
            rotation=None,
            rotation_axis="z",
            ensure_valid_placement=True,
            ensure_object_boundary_in_range=False,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.placement_initializer.append_sampler(frame_sampler)
        self.placement_initializer.append_sampler(parts_sampler)
        '''

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )
        
        # Track visual geoms for color reset
        self.obj_to_visual_geoms = {
            obj: obj.visual_geoms if hasattr(obj, "visual_geoms") else [] for obj in self.objects
        }

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj1_body_id = self.sim.model.body_name2id(self.obj1.root_body)
        self.obj2_body_id = self.sim.model.body_name2id(self.obj2.root_body)
        self.obj3_body_id = self.sim.model.body_name2id(self.obj3.root_body)
        self.obj4_body_id = self.sim.model.body_name2id(self.obj4.root_body)
        self.obj5_body_id = self.sim.model.body_name2id(self.obj5.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # obj1-related observables
            @sensor(modality=modality)
            def obj1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj1_body_id])

            @sensor(modality=modality)
            def obj1_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj1_body_id]), to="xyzw")

            # obj2-related observables
            @sensor(modality=modality)
            def obj2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj2_body_id])

            @sensor(modality=modality)
            def obj2_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj2_body_id]), to="xyzw")

            # obj3-related observables
            @sensor(modality=modality)
            def obj3_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj3_body_id])

            @sensor(modality=modality)
            def obj3_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj3_body_id]), to="xyzw")

            # obj4-related observables
            @sensor(modality=modality)
            def obj4_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj4_body_id])

            @sensor(modality=modality)
            def obj4_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj4_body_id]), to="xyzw")

            # obj5-related observables
            @sensor(modality=modality)
            def obj5_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj5_body_id])

            @sensor(modality=modality)
            def obj5_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj5_body_id]), to="xyzw")
                
            sensors = [obj1_pos, obj1_quat, obj2_pos, obj2_quat, obj3_pos, obj3_quat, obj4_pos, obj4_quat, obj5_pos, obj5_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, f"{obj}_pos", f"{arm_pf}gripper_to_{obj}", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                for obj in ["obj1", "obj2", "obj3", "obj4", "obj5"]
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        '''
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        '''
        
        # +-------------------------------------------------+
        # |                                                 |
        # |  obj2      obj3      obj4      obj5             |
        # |                                                 |
        # |                                 obj1 (frame)    |
        # |                                                 |
        # +-------------------------------------------------+
        # 			robot
        placements = {
            self.obj1: [-0.2, -0.15, self.table_offset[2] + 0.02], # [-0.2, -0.3, self.table_offset[2] -0.005] (before recentering .obj file)
            self.obj2: [0.12, -0.15, self.table_offset[2] + 0.045], # [0.1, -0.3, self.table_offset[2] -0.005] (before recentering .obj file)
            self.obj3: [0.12, 0.0, self.table_offset[2] + 0.0135], # [0.1, -0.15, self.table_offset[2] -0.005] (before recentering .obj file)
            self.obj4: [0.12, 0.15, self.table_offset[2] + 0.045], # [0.1, 0.0, self.table_offset[2] -0.005] (before recentering .obj file)
            self.obj5: [0.12, 0.3, self.table_offset[2] + 0.045], # [0.1, 0.15, self.table_offset[2] -0.005] (before recentering .obj file)
        }

        quat = np.array([1, 0, 0, 0])   # no rotation

        for obj, pos in placements.items():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(pos), quat])
            )

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cubeA)

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        reward = self.staged_rewards()
        return reward > 0
