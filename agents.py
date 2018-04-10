import numpy as np
import pandas as pd
from pysc2.lib import features, actions

# Screen features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# define constants about AI's world
_UNITS_MINE = 1
_UNITS_ENEMY = 4

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class Sc2Agent:
    def __init__(self, model):
        """
        :param model: Sc2Model object
        """
        self.model = model

    def act(self, obs):
        """
        Take a step and return an action using the policy
        :param obs: SC2Env state
        :return: SC2Action
        """
        return self.model.select_action(obs)

    def observe(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (state, action, reward, next_state)
        """
        for s, a, r, s_ in replay_buffer:
            self.model.update(s, a, r, s_)

    def obs_to_state(self, obs):
        return self.model.obs_to_state(obs)

    @property
    def training_mode(self):
        return self.model.training_mode

    @training_mode.setter
    def training_mode(self, val):
        self.model.training_mode = val
