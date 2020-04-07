import holodeck
import numpy as np
import imageio
import os
from collections import namedtuple

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


HolodeckObservation = namedtuple('HolodeckObservation', ['img', 'lin'])


class HolodeckEnv(Env):
    def __init__(self, scenario_name='', scenario_cfg=None, max_steps=200, 
                 gif_freq=500, steps_per_action=4, image_dir='images/test', 
                 viewport=False):

        # Load holodeck environment
        if scenario_cfg is not None and \
            scenario_cfg['package_name'] not in holodeck.installed_packages():
            
            holodeck.install(scenario_cfg['package_name'])

        self._env = holodeck.make(scenario_name=scenario_name, 
                                  scenario_cfg=scenario_cfg, 
                                  show_viewport=viewport)

        # Get action space from holodeck env and store for use with rlpyt
        if self.is_action_continuous:
            self._action_space = FloatBox(-1, 1, self._env.action_space.shape)

        else:
            self._action_space = IntBox(self._env.action_space.get_low(), 
                                        self._env.action_space.get_high(), 
                                        ())

        # Calculate observation space with all sensor data
        max_width = 0
        max_height = 0
        num_img = 0
        num_lin = 0
        for sensor in self._env._agent.sensors.values():
            if 'Task' in sensor.name:
                continue
            shape = sensor.sensor_data.shape
            if len(shape) == 3:
                max_width = max(max_width, shape[0])
                max_height = max(max_height, shape[1])
                num_img += shape[2]
            else:
                num_lin += np.prod(shape)
        
        if num_img > 0 and num_lin == 0:
            self.has_img = True
            self.has_lin = False
            self._observation_space = FloatBox(0, 1, 
                (num_img, max_width, max_height))
        elif num_lin > 0 and num_img == 0:
            self.has_img = False
            self.has_lin = True
            self._observation_space = FloatBox(-256, 256, (num_lin,))
        else:
            self.has_img = True
            self.has_lin = True
            self._observation_space = Composite([
                FloatBox(0, 1, (num_img, max_width, max_height)),
                FloatBox(-256, 256, (num_lin,))],
                HolodeckObservation)

        # Set data members
        self._max_steps = max_steps
        self._image_dir = image_dir
        self._steps_per_action = steps_per_action
        self.curr_step = 0
        self.gif_freq = gif_freq
        self.rollout_count = -1
        self.gif_images = []

    @property
    def horizon(self):
        return self._max_steps

    @property
    def img_size(self):
        img_null = self._get_img_null()
        if img_null is not None:
            return img_null.shape
        else:
            return None

    @property
    def lin_size(self):
        lin_null = self._get_lin_null()
        if lin_null is not None:
            return lin_null.shape[0]
        else:
            return None
        
    @property
    def action_size(self):
        return self._env.action_space.shape[0] \
            if self.is_action_continuous \
                else self._env.action_space.get_high()

    @property
    def is_action_continuous(self):
        return isinstance(self._env.action_space, 
                          holodeck.spaces.ContinuousActionSpace)

    def reset(self):
        ''' Resets env and returns initial state

            Returns:
                (np array)       
        '''
        sensor_dict = self._env.reset()
        self.curr_step = 0

        self.rollout_count += 1
        if len(self.gif_images) > 0:
            print('Making gif...')
            img_file = 'holodeck{}.gif'.format(self.rollout_count)
            img_path = os.path.join(self._image_dir, img_file)
            self._make_gif(self.gif_images, img_path)
            self.gif_images = []

        return self._get_state_rep(sensor_dict)

    def step(self, action):
        ''' Passes action to env and returns next state, reward, and terminal

            Args:
                action(int): Int represnting action in action space

            Returns:
                (EnvStep:named_tuple_array)        
        '''
        reward = 0

        if self.is_action_continuous: 
            action *= np.array(self._env.action_space.get_high())

        for _ in range(self._steps_per_action):
            sensor_dict, temp_reward, terminal, _ = self._env.step(action)
            reward += temp_reward

        if self.rollout_count % self.gif_freq == 0 and self.has_img:
            self.gif_images.append(self.get_img(sensor_dict))

        state_rep = self._get_state_rep(sensor_dict)

        self.curr_step += 1
        if self.curr_step >= self._max_steps:
            terminal = True

        return EnvStep(state_rep, np.array(reward), terminal, None)

    def get_img(self, sensor_dict):
        return sensor_dict['RGBCamera'] if self.has_img else None

    def _get_state_rep(self, sensor_dict):
        ''' Holodeck returns a dictionary of sensors. 
            The agent requires a single np array.

            Args:
                sensor_dict(dict(nparray)): A dictionay of array 
                representations of available sensors

            Returns:
                (nparray)
        '''

        if self._env.num_agents > 1:  # Only include main agent observations
            sensor_dict = sensor_dict[self._env._agent.name]  # TODO get main agent without accessing protected member

        img = self._get_img_null()
        lin = self._get_lin_null()

        img = img.astype(np.float32) if img is not None else None
        lin = lin.astype(np.float32) if lin is not None else None

        curr_img = 0
        curr_lin = 0
        for name, value in sensor_dict.items():
            if 'Task' in name:  # Do not include tasks in observation
                continue
            if len(value.shape) == 3:
                width = value.shape[0]
                height = value.shape[1]
                for c in range(value.shape[2]):
                    img[curr_img][:width][:height] = value[:,:,c] / 255
                    curr_img += 1
            else:
                sensor_flat = value.flatten()
                lin[curr_lin : curr_lin + sensor_flat.shape[0]] = sensor_flat
                curr_lin += sensor_flat.shape[0]

        if self.has_img and self.has_lin:
            return HolodeckObservation(img, lin)
        elif self.has_img:
            return img
        else:
            return lin

    def _make_gif(self, rollout, filename):
        with imageio.get_writer(filename, mode='I', duration=1 / 30) as writer:
            for x in rollout:
                writer.append_data((x).astype(np.uint8))

    def _get_img_null(self):
        if self.has_img and self.has_lin:
            return self._observation_space.spaces[0].null_value()
        elif self.has_img:
            return self._observation_space.null_value()
        else:
            return None

    def _get_lin_null(self):
        if self.has_img and self.has_lin:
            return self._observation_space.spaces[1].null_value()
        elif self.has_lin:
            return self._observation_space.null_value()
        else:
            return None
        