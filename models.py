import numpy as np
import pandas as pd
from pysc2.lib import actions

from sc2_env_functions import get_own_unit_location, get_enemy_unit_location
from sc_action import ScAction


class Sc2Model:
    def __init__(self, possible_actions):
        """
        :param possible_actions: list of ScAction objects
        """
        self._training_mode = True
        self.possible_actions = possible_actions

    def select_action(self, obs):
        return self._select_action(obs)

    def _select_action(self, obs):
        raise NotImplementedError("Please Implement this method")

    def update(self, s, a, r, s_):
        raise NotImplementedError("Please Implement this method")

    def obs_to_state(self, obs):
        raise NotImplementedError("Please Implement this method")

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, val):
        self._training_mode = val


class Sc2ScvBasedModel(Sc2Model):
    def select_action(self, obs):
        if not self.marine_selected(obs):
            return ScAction(actions.FUNCTIONS.no_op.id, has_location=False).get_function_call()
        return self._select_action(obs)

    def _select_action(self, obs):
        raise NotImplementedError("Please Implement this method")

    def update(self, s, a, r, s_):
        raise NotImplementedError("Please Implement this method")

    def obs_to_state(self, obs):
        return self.sc2obs_to_scv_relative_state(obs)

    @staticmethod
    def sc2obs_to_scv_relative_state(obs):
        """
        Convert sc2 obs object to a (dx, dy) state.
        :param obs: SC2Env observation
        :return:
        """
        marine_x, marine_y = get_own_unit_location(obs)
        scv_x, scv_y = get_enemy_unit_location(obs)

        dx = int(np.fabs(marine_x - scv_x))
        dy = int(np.fabs(marine_y - scv_y))

        return str((dx, dy))

    @staticmethod
    def marine_selected(obs):
        # For some reason the environment looses selection of my marine
        return actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']


class AlwayAttackScvModel(Sc2ScvBasedModel):
    def _select_action(self, obs):
        attack_scv_action = self.possible_actions[0]
        scv_x, scv_y = get_enemy_unit_location(obs)
        return attack_scv_action.get_function_call(location=(scv_x, scv_y))

    def update(self, s, a, r, s_):
        pass


class RandomModel(Sc2ScvBasedModel):
    def _select_action(self, obs):
        action = np.random.choice(self.possible_actions)
        scv_x, scv_y = get_enemy_unit_location(obs)
        return action.get_function_call(location=(scv_x, scv_y))

    def update(self, s, a, r, s_):
        pass


class QLearningTableScvFocusedModel(Sc2ScvBasedModel):
    """ Q learning table model where location is always on SCV """
    def __init__(self, possible_actions, learning_rate=0.01, reward_decay=0.9, epsilon_greedy=0.9):
        super().__init__(possible_actions)
        self.possible_actions_dict = {a.action_id: a for a in possible_actions}
        self.lr = learning_rate
        self.gamma = reward_decay
        self.init_epsilon = epsilon_greedy
        self.epsilon = epsilon_greedy
        self.q_table = pd.DataFrame(columns=self.possible_actions_dict.keys(), dtype=np.float64)

    def _select_action(self, obs):
        state = self.sc2obs_to_scv_relative_state(obs)
        self.check_state_exist(state)

        if not self.training_mode or np.random.uniform() < self.epsilon:
            # choose best action
            state_action_q_values = self.q_table.ix[state, :]

            # Shuffle to select randomly if some states are equal
            state_action_q_values = state_action_q_values.reindex(np.random.permutation(state_action_q_values.index))

            best_action_id_for_state = state_action_q_values.idxmax()
            action = self.possible_actions_dict[best_action_id_for_state]
        else:
            # choose random action
            action = np.random.choice(self.possible_actions)

        if action.has_location:
            scv_x, scv_y = get_enemy_unit_location(obs)
            return action.get_function_call(location=(scv_x, scv_y))
        else:
            return action.get_function_call()

    def update(self, s, a, r, s_):
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
                pd.Series([0] * len(self.possible_actions), index=self.q_table.columns, name=state))
