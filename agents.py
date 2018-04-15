import pickle
import numpy as np

from pysc2.lib import actions
from sc2_env_functions import get_own_unit_location, get_enemy_unit_location

from sc_action import ScAction
import constants


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
        if not self.marine_selected(obs):
            return ScAction(constants.NO_OP, actions.FUNCTIONS.no_op.id, has_location=False)
        return self._act(obs)

    def _act(self, obs):
        sc_action = self.model.select_action(obs)

        if sc_action.internal_id == constants.NO_OP:
            pass
        elif sc_action.internal_id == constants.ATTACK_ENEMY:
            location = get_enemy_unit_location(obs)
            sc_action.set_location(location)
        elif sc_action.internal_id == constants.MOVE_TO_ENEMY:
            location = get_enemy_unit_location(obs)
            sc_action.set_location(location)
        elif sc_action.internal_id == constants.MOVE_FROM_ENEMY:
            location = self.get_location_away(obs)
            sc_action.set_location(location)
        else:
            raise NotImplementedError("Unknown action ID received")

        return sc_action

    def observe(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (state, action, reward, next_state)
        """
        for s, a, r, s_ in replay_buffer:
            self.model.update(s, a, r, s_)

    def save(self, save_file='agent_file'):
        with open(save_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(save_file='agent_file'):
        with open(save_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        return tmp_dict

    @staticmethod
    def get_location_away(obs):
        screen_x_max = obs.observation['screen'].shape[1] - 1
        screen_y_max = obs.observation['screen'].shape[2] - 1

        own_location = get_own_unit_location(obs)
        enemy_location = get_enemy_unit_location(obs)

        # Distance between units
        dx = enemy_location[0] - own_location[0]
        dy = enemy_location[1] - own_location[1]

        # Move in opposite direction of enemy
        away_location = (own_location[0] - dx, own_location[1] - dy)

        # Makesure we don't move outside the screen
        away_location = (np.clip(away_location[0], 0, screen_x_max), np.clip(away_location[1], 0, screen_y_max))
        return away_location

    def obs_to_state(self, obs):
        return self.model.obs_to_state(obs)

    @property
    def training_mode(self):
        return self.model.training_mode

    @training_mode.setter
    def training_mode(self, val):
        self.model.training_mode = val

    @staticmethod
    def marine_selected(obs):
        # For some reason the environment looses selection of my marine
        return actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']
