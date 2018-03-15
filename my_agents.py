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


def get_marine_location(ai_relative_view):
    '''get the indices where the world is equal to 1'''
    return (ai_relative_view == _UNITS_MINE).nonzero()


def get_oponent_unit_location(ai_relative_view):
    '''get the indices where the world is equal to 1'''
    return (ai_relative_view == _UNITS_ENEMY).nonzero()


class MarineUpAgent(base_agent.BaseAgent):
    """An agent for doing a simple movement form one point to another."""

    def __init__(self):
        super().__init__()

    def step(self, obs):
        '''step function gets called automatically by pysc2 environment'''
        # call the parent class to have pysc2 setup rewards/etc
        super(MarineUpAgent, self).step(obs)

        # if self.steps < 100:
        #     return actions.FunctionCall(_NO_OP, [])

        # get what the ai can see about the world
        player_relative_screen = obs.observation['screen'][_PLAYER_RELATIVE]
        # get the location of our marine in this world
        marine_y, marine_x = get_marine_location(player_relative_screen)
        scv_y, scv_x = get_oponent_unit_location(player_relative_screen)


        time.sleep(1/5)

        available_action_names = helper.action_ids_to_action_names(obs.observation['available_actions'])
        # print(available_action_names)

        # if we can move our army (we have something selected)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            # it our marine is not on the screen do nothing.
            # this happens if we scroll away and look at a different
            # part of the world
            if not marine_x.any():
                return actions.FunctionCall(_NO_OP, [])
            # target = [marine_x.mean() + 1, marine_y.mean()]
            if scv_x.any():
                target = [scv_x.mean(), scv_y.mean()]
            else:
                target = [marine_x.mean() + 1, marine_y.mean()]

            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ADD])
            # return actions.FunctionCall(_SELECT_POINT, [[1], [marine_x.mean(), marine_y.mean()]])
