import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pysc2.lib import actions, features
from sc2_env_functions import get_own_unit_location, get_enemy_unit_location, get_enemy_width_and_height

from sc2_action import internal_id_to_action_id
import constants

OWN_PLAYER_FEATURE_ID = 1
ENEMY_PLAYER_FEATURE_ID = 2


class Sc2Agent:
    def __init__(self, model, log_dir, tb_train_writer, tb_test_writer):
        """
        :param model: Sc2Model object
        """
        self.tb_test_writer = tb_test_writer
        self.tb_train_writer = tb_train_writer
        self.log_dir = log_dir
        self.latest_replay_buffer = []
        self.episode_num = 0
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
            raise NotImplementedError(f"Unknown action ID {sc_action.internal_id} received")

        return sc_action

    def observe(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (state, action, reward, next_state)
        """
        self.episode_num += 1
        self.latest_replay_buffer = replay_buffer[:]

        if self.training_mode:
            random.shuffle(replay_buffer)
            return self.model.update(replay_buffer)
        else:
            # Update using an empty replay buffer because we don't want to update the model weights, only step it
            return self.model.update([])

    def save(self, save_folder):
        df = pd.DataFrame.from_records(self.latest_replay_buffer, columns=['state', 'action', 'reward', 'next_state'])
        df.to_csv(os.path.join(save_folder, 'replay_buffer.csv'))

        self.model.save(save_folder)

    @staticmethod
    def obs_to_state(obs):
        """
        Convert sc2 obs object to a distance_to_enemy state.
        :param obs: SC2Env observation
        :return:
        """
        own_unit_loc = np.array(get_own_unit_location(obs))
        enemy_loc = np.array(get_enemy_unit_location(obs))
        dist = np.linalg.norm(own_unit_loc - enemy_loc)
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
        # For some reason the environment looses selection of own unit from time to time
        return actions.FUNCTIONS.Move_screen.id in obs.observation['available_actions']

    def log_episode(self, last_obs, episode_number, results_dict):
        if self.training_mode:
            self._log_episode(self.tb_train_writer, last_obs, episode_number, results_dict)
        else:
            self._log_episode(self.tb_test_writer, last_obs, episode_number, results_dict)
            episode_log_folder = os.path.join(self.log_dir, str(episode_number))
            os.makedirs(episode_log_folder)
            self.save(episode_log_folder)

    @staticmethod
    def _log_episode(tb_writer, last_obs, episode_number, agent_results_dict):
        if tb_writer is None:
            return
        total_episode_rewards = last_obs.observation["score_cumulative"][0]
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode rewards', simple_value=total_episode_rewards)])
        tb_writer.add_summary(reward_summary, episode_number)

        if agent_results_dict is not None:
            for k, v in agent_results_dict.items():
                results_summary = tf.Summary(value=[tf.Summary.Value(tag=f'meta/{k}', simple_value=v)])
                tb_writer.add_summary(results_summary, episode_number)


class SimpleVikingAgent(Sc2Agent):
    def __init__(self, model, possible_actions, log_dir, tb_train_writer, tb_test_writer):
        super().__init__(model, log_dir, tb_train_writer, tb_test_writer)
        self.actions = possible_actions

    def _act(self, obs):
        state = self.obs_to_state(obs)
        illegal_internal_action_ids = self._get_illegal_internal_action_ids(obs)
        sc_action = self.model.select_action(state, illegal_internal_action_ids)

        if sc_action.internal_id == constants.NO_OP:
            pass
        elif sc_action.internal_id == constants.ATTACK_ENEMY:
            location = get_enemy_unit_location(obs)
            sc_action.set_location(location)
        elif sc_action.internal_id == constants.MOVE_TO_ENEMY:
            raise ValueError("Move to enemy action is illegal for this agent")
        elif sc_action.internal_id == constants.MOVE_FROM_ENEMY:
            location = self.get_location_away(obs)
            sc_action.set_location(location)
        elif sc_action.internal_id == constants.LAND:
            pass
        elif sc_action.internal_id == constants.FLIGHT:
            pass
        else:
            raise NotImplementedError("Unknown action ID received")

        return sc_action

    def obs_to_state(self, obs):
        """
        Convert sc2 obs object to a (distance_to_enemy, viking_flying_or_landed) state.
        :param obs: SC2Env observation
        :return:
        """
        # Distance
        own_unit_loc = np.array(get_own_unit_location(obs))
        enemy_loc = np.array(get_enemy_unit_location(obs))
        dist = np.linalg.norm(own_unit_loc - enemy_loc)
        rounded_dist = int(round(dist))

        # Flying or landed
        flying = None
        landed_action_id = internal_id_to_action_id(constants.LAND)
        can_land = landed_action_id in obs.observation['available_actions']
        if can_land:
            # If we can land we are flying
            flying = 'flying'
        else:
            flying_action_id = internal_id_to_action_id(constants.FLIGHT)
            can_fly = flying_action_id in obs.observation['available_actions']
            if can_fly:
                # If we can fly we are landed
                flying = 'landed'
            else:
                ValueError("Can neither land nor fly ... ")

        # Is enemy coming towards us
        enemy_closing_in = 'enemy_closing_in' if obs.reward > 0 and rounded_dist < 25 else 'enemy_not_attacking'

        return f'{rounded_dist}/{flying}/{enemy_closing_in}'

    def _get_illegal_internal_action_ids(self, obs):
        illegal_internal_action_id = []

        internal_and_action_ids = [(a.internal_id, internal_id_to_action_id(a.internal_id)) for a in self.actions]

        for internal_id, action_id in internal_and_action_ids:
            if action_id not in obs.observation['available_actions']:
                illegal_internal_action_id.append(internal_id)
        return illegal_internal_action_id

    def get_legal_internal_action_ids(self, obs):
        legal_internal_action_id = []

        internal_and_action_ids = [(a.internal_id, internal_id_to_action_id(a.internal_id)) for a in self.actions]

        for internal_id, action_id in internal_and_action_ids:
            if action_id in obs.observation['available_actions']:
                legal_internal_action_id.append(internal_id)
        return list(set(legal_internal_action_id))


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

    def save(self, save_file='agent_binary_file'):
        self.model.save(save_file)

    @staticmethod
    def load(save_file='agent_file'):
        pass
