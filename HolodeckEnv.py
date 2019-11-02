import holodeck
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


class HolodeckEnv(Env):
    def __init__(self, config, max_steps=200, use_cameras=True):
        self._env = holodeck.make(config)

        # Get action space from holodeck env and store for use with rlpyt
        if isinstance(self._env.action_space, holdeck.spaces.DiscreteActionSpace):
            self._action_space = IntBox(self._env.action_space._low, self._env.action_space._high, self._env.action_space.shape)
        elif isinstance(self._env.action_space, holdeck.spaces.ContinuousActionSpace):
            self._action_space = FloatBox(-10, 10, self._env.action_space.shape)

        # Calculate general observation space with all sensor data
        if use_cameras:
            max_width = 0
            max_height = 0
            num_dims = 0
            for sensor in self._env._agent.sensors.values():
                shape = sensor.shape
                if len(shape) > 2:
                    max_width = max(max_width, shape[0])
                    max_height = max(max_height, shape[1])
                    num_dims += shape[2]
            self._observation_space = FloatBox(0, 256, (max_width, max_height, num_dims))
        else:
            num_dims = 0
            for sensor in self._env._agent.sensors.values():
                shape = sensor.shape
                if len(shape) < 3:
                    num_dims += np.prod(shape)
            self._observation_space = FloatBox(-256, 256, (num_dims))

        self._max_steps = max_steps
        self.curr_step = 0
        self.use_cameras = use_cameras

    @property
    def horizon(self):
        return self._max_steps

    def reset(self):
        ''' Resets env and returns initial state

            Returns:
                (np array)       
        '''
        sensor_dict = self._env.reset()
        self.curr_step = 0
        return self.get_state_rep(sensor_dict)

    def step(self, action):
        ''' Passes action to env and returns next state, reward, and terminal

            Args:
                action(int): Int represnting action in action space

            Returns:
                (EnvStep:named_tuple_array)        
        '''
        sensor_dict, reward, terminal, _ = self._env.step(action)
        self.curr_step += 1
        if self.curr_step >= self._max_steps:
            terminal = True
        return EnvStep(self.get_state_rep(sensor_dict), reward, terminal, None)

    def get_state_rep(self, sensor_dict):
        ''' Holodeck returns a dictionary of sensors. 
            The agent requires a single np array.

            Args:
                sensor_dict(dict(nparray)): A dictionay of array representations of available sensors

            Returns:
                (nparray)
        '''
        state = self.observation_space.null_value()

        if use_cameras:
            curr_dim = 0
            for sensor in sensor_dict.values():
                if len(sensor.shape) > 2:
                    width = sensor.shape[0]
                    height = sensor.shape[1]
                    for c in sensor.shape[2]:
                        state[curr_dim][:width][:height] = sensor[c]
                        curr_dim += 1
        else:
            curr_dim = 0
            for sensor in sensor_dict.values():
                if len(sensor.shape) < 3:
                    flat = sensor.flatten()
                    state[curr_dim:curr_dim+flat.shape[0]] = flat
                    curr_dim += flat.shape[0]

        return state
