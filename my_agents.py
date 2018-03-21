import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import helper
import logging

import my_log
import helper

# define the features the AI can see
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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
GLOBAL_PARAM_MAX_EPISODES = None


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
        super(MyBaseAgent, self).step(obs)
        self.obs = obs

        # if self.steps < 100:
        #     return actions.FunctionCall(_NO_OP, [])

        # self.print_step_debug_data()

        if obs.reward > 0:
            r_per_frame = obs.reward / (self._steps_without_rewards * self._step_mul)
            msg = f'Reward after {self._steps_without_rewards} steps. R/Frames: {r_per_frame:.2f}'
            my_log.to_file(logging.INFO, msg)
            self._steps_until_rewards_array = np.append(self._steps_until_rewards_array, self._steps_without_rewards)
            self._steps_without_rewards = 0

            my_log.to_file(logging.INFO,f'Marine: {self._marine_location_per_step}')
            my_log.to_file(logging.INFO,f'SCV: {self._scv_location_per_step}')
            self._marine_location_per_step = []
            self._scv_location_per_step = []
            own_unit_loc, enemy_unit_loc = self._get_units_locations()
            self._marine_start_location.append(own_unit_loc)
            self._scv_start_location.append(enemy_unit_loc)
        else:
            self._steps_without_rewards += 1
            own_unit_loc, enemy_unit_loc = self._get_units_locations()
            self._marine_location_per_step.append((own_unit_loc, self._get_own_unit_locations()))
            self._scv_location_per_step.append((enemy_unit_loc, self._get_enemy_unit_locations()))

    def reset(self):
        super(MyBaseAgent, self).reset()

        if GLOBAL_PARAM_MAX_EPISODES and self.episodes > GLOBAL_PARAM_MAX_EPISODES:
            self._check_if_deterministic_agent()
            my_log.to_file(logging.WARNING, f'Average reward steps:{self._steps_until_rewards_array.mean()}')
            exit()  # There is no setting to quite after x episodes so  this is a hack for it.

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
        own_unit_loc_y, own_unit_loc_x = self._get_own_unit_locations()
        own_unit_loc = (own_unit_loc_x.mean(), own_unit_loc_y.mean())

        enemy_unit_loc_y, enemy_unit_loc_x = self._get_enemy_unit_locations()
        enemy_unit_loc = (enemy_unit_loc_x.mean(), enemy_unit_loc_y.mean())

        return own_unit_loc, enemy_unit_loc

    def _get_player_relative_view(self):
        """ View from player camera perspective. Returns an NxN np array """
        return self.obs.observation['screen'][_PLAYER_RELATIVE]

    def _get_own_unit_locations(self):
        """ My own unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_MINE).nonzero()

    def _get_enemy_unit_locations(self):
        """ Enemy unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_ENEMY).nonzero()

    def print_step_debug_data(self):
        enemy_unit_x, enemy_unit_y = self._get_enemy_unit_locations()
        print(f'Step {self.steps}, reward {self.obs.reward}, scv_alive {enemy_unit_x.any()}')

    def print_available_actions(self):
        print(helper.action_ids_to_action_names(self.obs.observation['available_actions']))


class AttackAlwaysAgent(MyBaseAgent):
    """Agent that attacks the enemy on every action"""

    def step(self, obs):
        super(AttackAlwaysAgent, self).step(obs)
        enemy_y, enemy_x = self._get_enemy_unit_locations()

        able_to_attack = _ATTACK_SCREEN in self.obs.observation['available_actions']
        if able_to_attack:
            target = [enemy_x.mean(), enemy_y.mean()]
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            # For some reason marine is not always able to attack the enemy...
            marine_y, marine_x = self._get_own_unit_locations()
            target = [marine_x.mean() + 1, marine_y.mean()]
            my_log.to_file(logging.INFO, 'Had to manually select marine')
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])


class AttackMoveAgent(MyBaseAgent):
    """Agent that alternates between attacking and moving towards enemy"""

    def step(self, obs):
        super(AttackMoveAgent, self).step(obs)
        enemy_y, enemy_x = self._get_enemy_unit_locations()

        able_to_attack = _ATTACK_SCREEN in self.obs.observation['available_actions']
        if able_to_attack:
            target = [enemy_x.mean(), enemy_y.mean()]
            action = _ATTACK_SCREEN if self.steps % 2 == 1 else _MOVE_SCREEN
            return actions.FunctionCall(action, [_NOT_QUEUED, target])
        else:
            # For some reason marine is not always able to attack the enemy...
            marine_y, marine_x = self._get_own_unit_locations()
            target = [marine_x.mean() + 1, marine_y.mean()]
            my_log.to_file(logging.INFO, 'Had to manually select marine')
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
