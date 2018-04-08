import numpy as np
from pysc2.lib import features, actions

# Screen features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# define constants about AI's world
_UNITS_MINE = 1
_UNITS_ENEMY = 4

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class BaseAgent:
    """ Base agent that uses (dx, dy) between marine and scv as state """
    def step(self, obs):
        """
        Take a step and return an action using the policy
        :param obs: SC2Env state
        :return: SC2Action
        """
        raise NotImplementedError()

    def update(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (obs, action, reward, next_obs)
        """
        raise NotImplementedError()

    def sc2obs_to_table_state(self, obs):
        """
        Convert sc2 obs object to a (dx, dy) state.
        :param obs: SC2Env state
        :return:
        """
        marine_x, marine_y = self._get_own_unit_location(obs)
        scv_x, scv_y = self._get_enemy_unit_location(obs)

        dx = int(np.fabs(marine_x - scv_x))
        dy = int(np.fabs(marine_y - scv_y))

        return dx, dy

    @staticmethod
    def marine_selected(obs):
        # For some reason the environment looses selection of my marine
        return actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']

    @staticmethod
    def _get_player_relative_view(obs):
        """ View from player camera perspective. Returns an NxN np array """
        return obs.observation["screen"][_PLAYER_RELATIVE]

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


class AlwaysAttackAgent(BaseAgent):
    def step(self, obs):
        if self.marine_selected(obs):
            scv_x, scv_y = self._get_enemy_unit_location(obs)
            return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [_NOT_QUEUED, (scv_x, scv_y)])
        else:
            return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])

    def update(self, replay_buffer):
        pass


class TableAgent(BaseAgent):
    ACTION_NO_OP = 0
    ACTION_ATTACK_ENEMY = 1
    ACTION_MOVE_TO_ENEMY = 2

    def __init__(self, x_size, y_size, step_size, discount):
        self.discount = discount
        self.step_size = step_size
        self.num_possible_actions = 3
        self.q_table = np.zeros(shape=(x_size, y_size, self.num_possible_actions))
        self.action_epsilon = 0.1

    def step(self, obs):
        if not self.marine_selected(obs):
            return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])

        do_random_action = np.random.rand() < self.action_epsilon
        if do_random_action:
            action_index = np.random.randint(self.num_possible_actions)
        else:
            dx, dy = self.sc2obs_to_table_state(obs)
            q_values = self.q_table[dx, dy, :]

            # TODO what happens if there is an argmax draw ?
            highest_action_value_for_state = np.argmax(q_values)
            action_index = highest_action_value_for_state

        if action_index == TableAgent.ACTION_NO_OP:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        elif action_index == TableAgent.ACTION_ATTACK_ENEMY:
            scv_x, scv_y = self._get_enemy_unit_location(obs)
            return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [_NOT_QUEUED, (scv_x, scv_y)])
        elif action_index == TableAgent.ACTION_MOVE_TO_ENEMY:
            scv_x, scv_y = self._get_enemy_unit_location(obs)
            return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [_NOT_QUEUED, (scv_x, scv_y)])
        else:
            raise Exception("Illegal action index!")

    def update(self, replay_buffer):
        print(f"IN UPDATE WITH BUFFER SIZE {len(replay_buffer)}")

