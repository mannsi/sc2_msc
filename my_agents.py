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


class MyBaseAgent(base_agent.BaseAgent):
    """My base agent"""

    def __init__(self):
        super().__init__()
        self.obs = None
        self._steps_without_rewards = 0
        self.step_mul = int(helper.get_command_param_val('--step_mul', remove_from_params=False))

    def step(self, obs):
        # call the parent class to have pysc2 setup rewards/etc
        super(MyBaseAgent, self).step(obs)
        self.obs = obs

        # if self.steps < 100:
        #     return actions.FunctionCall(_NO_OP, [])

        if self.steps == 1:
            self._log_units_location()

        if obs.reward > 0:
            r_per_frame = obs.reward / (self._steps_without_rewards * self.step_mul)
            msg = f'Reward {obs.reward} after {self._steps_without_rewards} steps. R/Frames: {r_per_frame:.2f}'
            my_log.to_file(logging.INFO, msg)
            self._steps_without_rewards = 0
            self._log_units_location()
        else:
            self._steps_without_rewards += 1

    def _log_units_location(self):
        own_unit_start_loc_y, own_unit_start_loc_x = self._get_own_unit_locations()
        own_unit_start_loc = (own_unit_start_loc_x.mean(), own_unit_start_loc_y.mean())

        enemy_unit_start_loc_y, enemy_unit_start_loc_x = self._get_enemy_unit_locations()
        enemy_unit_start_loc = (enemy_unit_start_loc_x.mean(), enemy_unit_start_loc_y.mean())

        my_log.to_file(logging.DEBUG, f'Marine loc: {own_unit_start_loc}, Enemy loc: {enemy_unit_start_loc}')

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
        self.print_step_debug_data()

        enemy_y, enemy_x = self._get_enemy_unit_locations()
        enemy_found = enemy_x.any()

        able_to_attack = _ATTACK_SCREEN in self.obs.observation['available_actions']
        if enemy_found and able_to_attack:
            target = [enemy_x.mean(), enemy_y.mean()]
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            # Hacky code since enemy sometimes moves out of screen range
            marine_y, marine_x = self._get_own_unit_locations()
            target = [marine_x.mean() + 1, marine_y.mean()]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])


class AttackMoveAgent(MyBaseAgent):
    """Agent that alternates between attacking and moving towards enemy"""

    def step(self, obs):
        super(AttackMoveAgent, self).step(obs)

        # time.sleep(1/5)
        # self.print_step_debug_data()

        enemy_y, enemy_x = self._get_enemy_unit_locations()
        enemy_found = enemy_x.any()
        if enemy_found:
            target = [enemy_x.mean(), enemy_y.mean()]
        else:
            # Hacky code since enemy sometimes moves out of screen range
            marine_y, marine_x = self._get_own_unit_locations()
            target = [marine_x.mean() + 1, marine_y.mean()]

        action = _ATTACK_SCREEN if self.steps % 2 == 1 else _MOVE_SCREEN
        return actions.FunctionCall(action, [_NOT_QUEUED, target])