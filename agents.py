import pickle
import numpy as np

from pysc2.lib import actions, features
from sc2_env_functions import get_own_unit_location, get_enemy_unit_location, get_enemy_width_and_height

from sc2_action import Sc2Action
import constants

OWN_PLAYER_FEATURE_ID = 1
ENEMY_PLAYER_FEATURE_ID = 2


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
        if not self.own_units_selected(obs):
            return self.model.default_action()
        return self._act(obs)

    def _act(self, obs):
        state = self.obs_to_state(obs)
        sc_action = self.model.select_action(state)

        if sc_action.internal_id == constants.NO_OP:
            pass
        elif sc_action.internal_id == constants.ATTACK_ENEMY:
            location = get_enemy_unit_location(obs)
            sc_action.set_location(location)
        elif sc_action.internal_id == constants.MOVE_TO_ENEMY:
            location = self.get_location_to(obs)

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
        return self.model.update(replay_buffer)

    def save(self, save_file='agent_file'):
        with open(save_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(save_file='agent_file'):
        with open(save_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        return tmp_dict

    @staticmethod
    def obs_to_state(obs):
        """
        Convert sc2 obs object to a distance_to_enemy state.
        :param obs: SC2Env observation
        :return:
        """
        marine_loc = np.array(get_own_unit_location(obs))
        enemy_loc = np.array(get_enemy_unit_location(obs))
        dist = np.linalg.norm(marine_loc - enemy_loc)
        rounded_dist = int(round(dist))
        return rounded_dist

    def get_location_away(self, obs):
        own_location = get_own_unit_location(obs)
        enemy_location = get_enemy_unit_location(obs)

        # Distance between units
        dx = enemy_location[0] - own_location[0]
        dy = enemy_location[1] - own_location[1]

        # Move in opposite direction of enemy
        away_location = (own_location[0] - dx, own_location[1] - dy)

        # Make sure we don't move outside the screen
        away_location = self.clip_location(away_location, obs)
        return away_location

    def get_location_to(self, obs):
        """
        Get the location we want to move towards the enemy. We cannot move 'on' the enemy because that equals attacking
        :return: x, y
        """
        enemy_x_min, enemy_x_max, enemy_y_min, enemy_y_max = get_enemy_width_and_height(obs)
        own_unit_x, own_unit_y = get_own_unit_location(obs)
        enemy_x, enemy_y = get_enemy_unit_location(obs)

        # Select between 4 quadrants below/above and left/right of the enemy to move to.
        # We want to move further than the enemy in case we are standing too close to move to a location between our
        # unit(s) and the enemy

        if enemy_x >= own_unit_x:
            x = enemy_x_max + 2
        else:
            x = enemy_x_min - 2

        if enemy_y >= own_unit_y:
            y = enemy_y_max + 2
        else:
            y = enemy_y_min - 2

        location_to = self.clip_location((x, y), obs)
        return location_to

    @staticmethod
    def clip_location(location, obs):
        """
        Returns a clipped location so as to not move outside the allowed screen coordinates
        :param location: (x, y)
        :param obs: Sc2env observation
        :return: x, y
        """
        screen_x_max = obs.observation['screen'].shape[1] - 1
        screen_y_max = obs.observation['screen'].shape[2] - 1

        clipped_location = (np.clip(location[0], 0, screen_x_max), np.clip(location[1], 0, screen_y_max))
        return clipped_location

    @property
    def training_mode(self):
        return self.model.training_mode

    @training_mode.setter
    def training_mode(self, val):
        self.model.training_mode = val

    @staticmethod
    def own_units_selected(obs):
        # For some reason the environment looses selection of my marine
        return actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']


class Simple1DAgent(Sc2Agent):
    def obs_to_state(self, obs):
        """
        Convert sc2 obs object to a distance_to_enemy state.
        :param obs: SC2Env observation
        :return:
        """
        # Which player owns which units
        player_id_feature = obs.observation['screen'][features.SCREEN_FEATURES.player_id.index]

        own_units_feature = np.array(player_id_feature == OWN_PLAYER_FEATURE_ID, dtype=int)
        enemy_units_feature = np.array(player_id_feature == ENEMY_PLAYER_FEATURE_ID, dtype=int)

        all_features = np.stack((own_units_feature, enemy_units_feature))
        state = all_features.flatten()
        return state.reshape(1, state.shape[0])

    def save(self, save_file='agent_file'):
        self.model.save(save_file)

    @staticmethod
    def load(save_file='agent_file'):
        pass


class Simple2DAgent(Sc2Agent):
    def obs_to_state(self, obs):
        """
        Convert sc2 obs object to a distance_to_enemy state.
        :param obs: SC2Env observation
        :return:
        """
        # Which player owns which units
        player_id_feature = obs.observation['screen'][features.SCREEN_FEATURES.player_id.index]

        own_units_feature = np.array(player_id_feature == OWN_PLAYER_FEATURE_ID, dtype=int)
        enemy_units_feature = np.array(player_id_feature == ENEMY_PLAYER_FEATURE_ID, dtype=int)

        all_features = np.stack((own_units_feature, enemy_units_feature))

        return all_features