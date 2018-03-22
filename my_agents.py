import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import logging

import my_log
import helper
import my_plotting

# Screen features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index

# define contstants for actions
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id


_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

# define constants about AI's world
_UNITS_MINE = 1
_UNITS_ENEMY = 4

# constants for actions
_SELECT_ADD = [1]
_NOT_QUEUED = [0]


# Stupid way to set parameters because I get errors from absl library if I add a flag it does not recognize
GLOBAL_PARAM_MAX_EPISODES = -1
GLOBAL_WAIT_AFTER_ATTACK = -1
GLOBAL_MOVE_STEPS_AFTER_ATTACK = -1


class MyBaseAgent(base_agent.BaseAgent):
    """My base agent"""

    def __init__(self):
        super().__init__()
        self.obs = None
        self._steps_without_rewards = 0
        self._step_mul = int(helper.get_command_param_val('--step_mul', remove_from_params=False, default_val=8))
        self._steps_until_rewards_array = np.array([])
        self._marine_location_per_step = []
        self._scv_location_per_step = []
        self._marine_start_location = []
        self._scv_start_location = []

    def step(self, obs):
        # call the parent class to have pysc2 setup rewards/etc
        super().step(obs)
        self.obs = obs

        # if self.steps < 100:
        #     return actions.FunctionCall(_NO_OP, [])

        # self.print_step_debug_data()

        if obs.reward > 0:
            self.got_reward(obs.reward)
        else:
            self._steps_without_rewards += 1
            own_unit_loc, enemy_unit_loc = self._get_units_locations()
            self._marine_location_per_step.append((own_unit_loc, self._get_own_unit_locations()))
            self._scv_location_per_step.append((enemy_unit_loc, self._get_enemy_unit_locations()))

    def reset(self):
        super().reset()

        if GLOBAL_PARAM_MAX_EPISODES and self.episodes > GLOBAL_PARAM_MAX_EPISODES:
            self._check_if_deterministic_agent()
            my_log.to_file(logging.INFO, f'Average reward steps:{self._steps_until_rewards_array.mean()}')
            exit()  # There is no setting to quite after x episodes so  this is a hack for it.

    def got_reward(self, reward):
        r_per_frame = reward / (self._steps_without_rewards * self._step_mul)
        msg = f'Reward after {self._steps_without_rewards} steps. R/Frames: {r_per_frame:.2f}'
        my_log.to_file(logging.INFO, msg)
        self._steps_until_rewards_array = np.append(self._steps_until_rewards_array, self._steps_without_rewards)
        self._steps_without_rewards = 0

        my_log.to_file(logging.INFO, f'Marine: {self._marine_location_per_step}')
        my_log.to_file(logging.INFO, f'SCV: {self._scv_location_per_step}')
        self._marine_location_per_step = []
        self._scv_location_per_step = []
        own_unit_loc, enemy_unit_loc = self._get_units_locations()
        self._marine_start_location.append(own_unit_loc)
        self._scv_start_location.append(enemy_unit_loc)

    def _check_if_deterministic_agent(self):
        # Unable to confirm number of steps until scv dies but there is cool down env randomness for marine attack
        # So instead we check if units always start roughly at the same location.
        location_threshold = 2  # Some bloody random noise in the env that means units do now always spawn at same pxl

        max_marine_x = max([x[0] for x in self._marine_start_location])
        max_marine_y = max([x[1] for x in self._marine_start_location])
        min_marine_x = min([x[0] for x in self._marine_start_location])
        min_marine_y = min([x[1] for x in self._marine_start_location])
        max_scv_x = max([x[0] for x in self._scv_start_location])
        max_scv_y = max([x[1] for x in self._scv_start_location])
        min_scv_x = min([x[0] for x in self._scv_start_location])
        min_scv_y = min([x[1] for x in self._scv_start_location])

        if (max_marine_x - min_marine_x) > location_threshold:
            my_log.to_file(logging.INFO, f'NONDETER! Marine_max_x: {max_marine_x}, Marine_min_x: {min_marine_x}')
        elif (max_marine_y - min_marine_y) > location_threshold:
            my_log.to_file(logging.INFO, f'NONDETER! Marine_max_y: {max_marine_y}, Marine_min_y: {min_marine_y}')
        elif (max_scv_x - min_scv_x) > location_threshold:
            my_log.to_file(logging.INFO, f'NONDETER! scv_max_x: {max_scv_x}, scv_min_x: {min_scv_x}')
        elif (max_scv_y - min_scv_y) > location_threshold:
            my_log.to_file(logging.INFO, f'NONDETER! scv_max_y: {max_scv_y}, scv_min_y: {min_scv_y}')
        else:
            my_log.to_file(logging.INFO, f'DETERMINISTIC AGENT!')

    def _get_units_locations(self):
        own_unit_loc = self._get_own_unit_location()
        enemy_unit_loc = self._get_enemy_unit_location()
        return own_unit_loc, enemy_unit_loc

    def _get_player_relative_view(self):
        """ View from player camera perspective. Returns an NxN np array """
        return self.obs.observation["feature_screen"][_PLAYER_RELATIVE]

    def _get_own_unit_location(self):
        """ Mean values of friendly unit coordinates, returned as a (x,y) tuple """
        own_unit_loc_y, own_unit_loc_x = self._get_own_unit_locations()
        return own_unit_loc_x.mean(), own_unit_loc_y.mean()

    def _get_own_unit_locations(self):
        """ My own unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_MINE).nonzero()

    def _get_enemy_unit_location(self):
        """ Mean values of enemy unit coordinates, returned as a (x,y) tuple """
        enemy_unit_loc_y, enemy_unit_loc_x = self._get_enemy_unit_locations()
        return enemy_unit_loc_x.mean(), enemy_unit_loc_y.mean()

    def _get_enemy_unit_locations(self):
        """ Enemy unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_ENEMY).nonzero()

    def _attack_enemy_action(self):
        target = self._get_enemy_unit_location()
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])

    def _move_to_enemy_action(self):
        target = self._get_enemy_unit_location()
        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

    def _select_own_unit(self):
        target = self._get_own_unit_location()
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    def _print_step_debug_data(self):
        enemy_unit_x, enemy_unit_y = self._get_enemy_unit_locations()
        print(f'Step {self.steps}, reward {self.obs.reward}, scv_alive {enemy_unit_x.any()}')

    def _print_available_actions(self):
        print(helper.action_ids_to_action_names(self.obs.observation['available_actions']))


class AttackAlwaysAgent(MyBaseAgent):
    """Agent that attacks the enemy on every action"""

    def step(self, obs):
        super().step(obs)

        # For some reason the environment looses selection of my marine
        if _ATTACK_SCREEN not in self.obs.observation['available_actions']:
            my_log.to_file(logging.INFO, f'Unable to attack. Step {self.steps}')
            return self._select_own_unit()

        return self._attack_enemy_action()


class AttackMoveAgent(MyBaseAgent):
    """Agent that alternates between attacking and moving towards enemy"""

    def step(self, obs):
        super().step(obs)

        # For some reason the environment looses selection of my marine
        if _ATTACK_SCREEN not in self.obs.observation['available_actions']:
            my_log.to_file(logging.INFO, f'Unable to attack. Step {self.steps}')
            return self._select_own_unit()

        if self.steps % 2 == 1:
            return self._attack_enemy_action()
        else:
            return self._move_to_enemy_action()


class DiscoverStepsAgent(MyBaseAgent):
    """Agent that maps how many steps it takes to damage the SCV"""

    def __init__(self):
        super().__init__()
        self._scv_last_health = 45
        self._steps_since_last_damage = 0
        self.steps_between_dmg_list = []
        self.scv_has_taken_first_dmg = False

    def step(self, obs):
        super().step(obs)

        # For some reason the environment looses selection of my marine
        if _ATTACK_SCREEN not in self.obs.observation['available_actions']:
            my_log.to_file(logging.INFO, f'Unable to attack. Step {self.steps}')
            return self._select_own_unit()

        scv_health = self.get_scv_health()
        if scv_health < self._scv_last_health:
            self.steps_between_dmg_list.append(self._steps_since_last_damage)
            self._scv_last_health = scv_health
            self._steps_since_last_damage = 0
            self.scv_has_taken_first_dmg = True
        elif self.scv_has_taken_first_dmg:
            self._steps_since_last_damage += 1

        return self._attack_enemy_action()

    def get_scv_health(self):
        scv_location = self._get_enemy_unit_location()
        return self._get_unit_health(scv_location)

    def _get_unit_health(self, location):
        return self.obs.observation["feature_screen"][_UNIT_HIT_POINTS][int(location[1]), int(location[0])]


class DiscoverSsmAgent(DiscoverStepsAgent):
    """ Agent that tries to discover move/attack ratios to achieve stutter step micro """

    def __init__(self):
        super().__init__()
        self.steps_between_dmg_list_of_lists = []
        self.steps_to_kill_scv = []

    def step(self, obs):
        super().step(obs)

        # For some reason the environment looses selection of my marine
        if _ATTACK_SCREEN not in self.obs.observation['available_actions']:
            return self._select_own_unit()

        if GLOBAL_WAIT_AFTER_ATTACK <= self._steps_since_last_damage  and \
                (self._steps_since_last_damage - GLOBAL_WAIT_AFTER_ATTACK) < GLOBAL_MOVE_STEPS_AFTER_ATTACK:
            return self._move_to_enemy_action()

        return self._attack_enemy_action()

    def reset(self):
        # Needs to be above super because super kills the run before we are able to log.
        self.steps_between_dmg_list = []
        if GLOBAL_PARAM_MAX_EPISODES and self.episodes == GLOBAL_PARAM_MAX_EPISODES:
            np_dmg_steps_means = np.array(self.steps_between_dmg_list_of_lists).mean(axis=0)
            my_log.to_file(logging.WARNING, f'Average steps between damage are {np_dmg_steps_means}')

            avg_steps_to_kill = np.array(self.steps_to_kill_scv).mean()

            my_plotting.save_results_to_file(
                'ssm_move_steps.txt',
                GLOBAL_WAIT_AFTER_ATTACK,
                GLOBAL_MOVE_STEPS_AFTER_ATTACK,
                np_dmg_steps_means.mean(),
                avg_steps_to_kill
            )

        super().reset()

    def got_reward(self, reward):
        super().got_reward(reward)
        self.steps_between_dmg_list_of_lists.append(self.steps_between_dmg_list)
        first_reward = len(self.steps_to_kill_scv) == 0
        if first_reward:
            self.steps_to_kill_scv.append(self.steps)
        else:
            last_reward_step = self.steps_to_kill_scv[-1]
            self.steps_to_kill_scv.append(self.steps - last_reward_step)

