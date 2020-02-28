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


# TODO forest: make reward distance=1
class HolodeckEnv(Env):
    def __init__(self, scenario_name='', scenario_cfg=None, max_steps=200, 
                 gif_freq=500, image_dir='images/test', viewport=False):

        self._env = holodeck.make(scenario_name=scenario_name, 
                                  scenario_cfg=scenario_cfg, 
                                  show_viewport=viewport)  # TODO fix remote viewportfalse error

        # Get action space from holodeck env and store for use with rlpyt
        if self.is_action_continuous:
            self._action_space = FloatBox(-1, 1, self._env.action_space.shape)

        else:
            self._action_space = IntBox(self._env.action_space._low, 
                                        self._env.action_space._high, 
                                        self._env.action_space.shape)  # TODO don't access protected data members

        # TODO fix sphere control scheme
        self._env.set_control_scheme('sphere0', 1)  
        self._action_space = IntBox(0, 4, ())

        # Calculate general observation space with all sensor data
        max_width = 0
        max_height = 0
        num_img = 0
        num_lin = 0
        for sensor in self._env._agent.sensors.values():
            shape = sensor.sensor_data.shape
            if len(shape) == 3:
                max_width = max(max_width, shape[0])
                max_height = max(max_height, shape[1])
                num_img += shape[2]
            else:
                num_lin += np.prod(shape)
        
        self._observation_space = Composite([
            FloatBox(0, 1, (num_img, max_width, max_height)),
            FloatBox(-256, 256, (num_lin,))],
            HolodeckObservation)

        self._max_steps = max_steps
        self._image_dir = image_dir
        self.curr_step = 0
        self.gif_freq = gif_freq
        self.rollout_count = 0
        self.gif_images = []

    @property
    def horizon(self):
        return self._max_steps

    @property
    def img_size(self):
        return self._observation_space.spaces[0].null_value().shape

    @property
    def lin_size(self):
        return self._observation_space.spaces[1].null_value().shape[0]

    @property
    def action_size(self):
        return self._env.action_space.shape[0] \
            if self.is_action_continuous \
                else self._env.action_space._high

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
        for _ in range(4):  # TODO make this part of holodeck
            sensor_dict, temp_reward, terminal, _ = self._env.step(action)  # * np.array([6.508, 5.087, .8, 59.844]))  # TODO get bounds programatically
            reward += temp_reward

        if self.rollout_count % self.gif_freq == 0:
            self.gif_images.append(self.get_img(sensor_dict))

        state_rep = self._get_state_rep(sensor_dict)

        self.curr_step += 1
        if self.curr_step >= self._max_steps:
            terminal = True

        return EnvStep(state_rep, np.array(reward), terminal, None)

    def get_img(self, sensor_dict):
        return sensor_dict['RGBCamera']

    def _get_state_rep(self, sensor_dict):
        ''' Holodeck returns a dictionary of sensors. 
            The agent requires a single np array.

            Args:
                sensor_dict(dict(nparray)): A dictionay of array 
                representations of available sensors

            Returns:
                (nparray)
        '''

        img = self._observation_space.spaces[0].null_value().astype(np.float32)
        lin = self._observation_space.spaces[1].null_value().astype(np.float32)

        curr_img = 0
        curr_lin = 0
        for sensor in sensor_dict.values():
            if len(sensor.shape) == 3:
                width = sensor.shape[0]
                height = sensor.shape[1]
                for c in range(sensor.shape[2]):
                    img[curr_img][:width][:height] = sensor[:,:,c] / 255
                    curr_img += 1
            else:
                sensor_flat = sensor.flatten()
                lin[curr_lin : curr_lin + sensor_flat.shape[0]] = sensor_flat
                curr_lin += sensor_flat.shape[0]

        return HolodeckObservation(img, lin)

    def _make_gif(self, rollout, filename):
        with imageio.get_writer(filename, mode='I', duration=1 / 30) as writer:
            for x in rollout:
                writer.append_data(((x[0, :, :] + 0.5) * 255).astype(np.uint8))
