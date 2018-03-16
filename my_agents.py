from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
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

    def step(self, obs):
        '''step function gets called automatically by pysc2 environment'''
        # call the parent class to have pysc2 setup rewards/etc
        super(MyBaseAgent, self).step(obs)
        self.obs = obs

    def _get_player_relative_view(self):
        """ View from player camera perspective. Returns an NxN np array """
        return self.obs.observation['screen'][_PLAYER_RELATIVE]

    def _get_own_unit_locations(self):
        """ My own unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_MINE).nonzero()

    def _get_oponent_unit_locations(self):
        """ Enemy unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view() == _UNITS_ENEMY).nonzero()

    def print_step_debug_data(self):
        enemy_unit_x, enemy_unit_y = self._get_oponent_unit_locations()
        print(f'Step {self.steps}, reward {self.obs.reward}, scv_alive {enemy_unit_x.any()}')

    def print_available_actions(self):
        print(helper.action_ids_to_action_names(self.obs.observation['available_actions']))


class AttackAlwaysAgent(MyBaseAgent):
    """Agent that attacks the enemy on every action"""

    def step(self, obs):
        '''step function gets called automatically by pysc2 environment'''
        # call the parent class to have pysc2 setup rewards/etc
        super(AttackAlwaysAgent, self).step(obs)

        # if self.steps < 100:
        #     return actions.FunctionCall(_NO_OP, [])

        # time.sleep(1/5)
        self.print_step_debug_data()

        scv_y, scv_x = self._get_oponent_unit_locations()
        scv_found = scv_x.any()
        if scv_found:
            target = [scv_x.mean(), scv_y.mean()]
        else:
            marine_y, marine_x = self._get_own_unit_locations()
            target = [marine_x.mean() + 1, marine_y.mean()]
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])

