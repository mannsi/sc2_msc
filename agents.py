import numpy as np
import pandas as pd
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
    def __init__(self):
        self.is_training = True

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

    @property
    def training_mode(self):
        return self.is_training

    @training_mode.setter
    def training_mode(self, val):
        self.is_training = val

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

        return str((dx, dy))

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
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def update(self, replay_buffer):
        pass


class TableAgent(BaseAgent):
    def __init__(self, sc_actions, learning_rate, reward_decay, epsilon_greedy):
        """
        :param sc_actions: list[ScAction]
        :param learning_rate:
        :param reward_decay:
        :param epsilon_greedy:
        """
        super().__init__()
        self.sc_actions = sc_actions
        sc_action_ids = [a.id for a in sc_actions]
        self.q_table = QLearningTable(sc_action_ids, learning_rate, reward_decay, epsilon_greedy)
        self.epsilon = 0.1

    def step(self, obs):
        if not self.marine_selected(obs):
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

        state = self.sc2obs_to_table_state(obs)
        selected_action_id = self.q_table.choose_action(state, self.is_training)
        selected_action = [a for a in self.sc_actions if a.id == selected_action_id][0]

        if selected_action.req_location:
            scv_x, scv_y = self._get_enemy_unit_location(obs)
            action_param = [_NOT_QUEUED, (scv_x, scv_y)]
        else:
            action_param = []
        return actions.FunctionCall(selected_action.id, action_param)

    def update(self, replay_buffer):
        for state, action, reward, next_state in replay_buffer:
            self.q_table.learn(state, action, reward, next_state)


class QLearningTable:
    def __init__(self, action_ids, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.action_ids = action_ids
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.action_ids, dtype=np.float64)

    def choose_action(self, state, allow_random):
        self.check_state_exist(state)

        if not allow_random or np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[state, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.action_ids)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.action_ids), index=self.q_table.columns, name=state))


class ScAction:
    def __init__(self, id, req_location):
        self.id = id
        self.req_location = req_location
