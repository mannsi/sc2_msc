import numpy as np
from pysc2.lib import features, actions

# Screen features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# define constants about AI's world
_UNITS_MINE = 1
_UNITS_ENEMY = 4

_NOT_QUEUED = [0]

_ACTION_NO_OP = 0
_ACTION_ATTACK_ENEMY = 1
_ACTION_MOVE_TO_ENEMY = 2


class TableAgent:
    def __init__(self, x_size, y_size, step_size, discount):
        self.discount = discount
        self.step_size = step_size
        self.num_possible_actions = 3
        self.q_table = np.zeros(shape=(x_size, y_size, self.num_possible_actions))
        self.action_epsilon = 0.1

    def step(self, obs):
        """
        Take a step and return an action using the policy
        :param obs: SC2Env state
        :return: SC2Action
        """
        state = self.sc2obs_to_table_state(obs)
        (x, y) = state

        do_random_action = np.random.rand() < self.action_epsilon
        if do_random_action:
            action_index = np.random.randint(self.num_possible_actions)
        else:
            q_values = self.q_table[x, y, :]
            highest_action_value_for_state = np.argmax(q_values)
            action_index = highest_action_value_for_state

        if action_index == _ACTION_NO_OP:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        elif action_index == _ACTION_ATTACK_ENEMY:
            actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [_NOT_QUEUED, (x, y)])
        elif action_index == _ACTION_MOVE_TO_ENEMY:
            actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [_NOT_QUEUED, (x, y)])
        else:
            raise Exception("Illegal action index!")

    def update(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (state, action, reward, next_state)
        :return:
        """
        pass

    def sc2obs_to_table_state(self, obs):
        """
        Convert sc2 obs object to state that makes sense for the Q table agent.
        :param obs: SC2Env state
        :return:
        """
        marine_x, marine_y = self._get_own_unit_location(obs)
        scv_x, scv_y = self._get_enemy_unit_location(obs)

        dx = int(np.fabs(marine_x - scv_x))
        dy = int(np.fabs(marine_y - scv_y))

        return dx, dy

    @staticmethod
    def _get_player_relative_view(obs):
        """ View from player camera perspective. Returns an NxN np array """
        return obs.observation["feature_screen"][_PLAYER_RELATIVE]

    def _get_own_unit_location(self, obs):
        """ Mean values of friendly unit coordinates, returned as a (x,y) tuple """
        own_unit_loc_y, own_unit_loc_x = self._get_own_unit_locations(obs)
        return own_unit_loc_x.mean(), own_unit_loc_y.mean()

    def _get_own_unit_locations(self, obs):
        """ My own unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view(obs) == _UNITS_MINE).nonzero()

    def _get_enemy_unit_location(self, obs):
        """ Mean values of enemy unit coordinates, returned as a (x,y) tuple """
        enemy_unit_loc_y, enemy_unit_loc_x = self._get_enemy_unit_locations(obs)
        return enemy_unit_loc_x.mean(), enemy_unit_loc_y.mean()

    def _get_enemy_unit_locations(self, obs):
        """ Enemy unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
        return (self._get_player_relative_view(obs) == _UNITS_ENEMY).nonzero()
