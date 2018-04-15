import numpy as np
import pandas as pd
import pickle

from sc2_env_functions import get_own_unit_location, get_enemy_unit_location


class Sc2Model:
    def __init__(self, possible_actions):
        """
        :param possible_actions: list of ScAction objects
        """
        self._training_mode = True
        self.possible_actions = possible_actions

    def select_action(self, obs):
        """
                Select an action given an observation
                :param obs: sc2env observation
                :return: sc_action.ScAction object
                """
        raise NotImplementedError("Please Implement this method")

    def update(self, s, a, r, s_):
        raise NotImplementedError("Please Implement this method")

    def obs_to_state(self, obs):
        """
        Convert sc2 obs object to a distance_to_enemy state.
        :param obs: SC2Env observation
        :return:
        """
        marine_loc = np.array(get_own_unit_location(obs))
        enemy_loc = np.array(get_enemy_unit_location(obs))
        dist = np.linalg.norm(marine_loc - enemy_loc)
        return str(dist)

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, val):
        self._training_mode = val


class AlwayAttackEnemyModel(Sc2Model):
    def select_action(self, obs):
        return self.possible_actions[0]

    def update(self, s, a, r, s_):
        pass


class RandomModel(Sc2Model):
    def select_action(self, obs):
        return np.random.choice(self.possible_actions)

    def update(self, s, a, r, s_):
        pass


class QLearningTableEnemyFocusedModel(Sc2Model):
    """ Q learning table model where location is always on SCV """
    def __init__(self, possible_actions, learning_rate=0.01, reward_decay=0.9, epsilon_greedy=0.9):
        super().__init__(possible_actions)
        self.possible_actions_dict = {a.internal_id: a for a in possible_actions}
        self.lr = learning_rate
        self.gamma = reward_decay
        self.init_epsilon = epsilon_greedy
        self.epsilon = epsilon_greedy
        self.q_table = pd.DataFrame(columns=self.possible_actions_dict.keys(), dtype=np.float64)

    def select_action(self, obs):
        state = self.obs_to_state(obs)
        self.check_state_exist(state)

        if not self.training_mode or np.random.uniform() < self.epsilon:
            # choose best action
            state_action_q_values = self.q_table.ix[state, :]

            # Shuffle to select randomly if some states are equal
            state_action_q_values = state_action_q_values.reindex(np.random.permutation(state_action_q_values.index))

            best_internal_action_id_for_state = state_action_q_values.idxmax()
            action = self.possible_actions_dict[best_internal_action_id_for_state]
        else:
            # choose random action
            action = np.random.choice(self.possible_actions)

        return action

    def update(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a.internal_id]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a.internal_id] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            ser = pd.Series([0] * len(self.possible_actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(ser)
