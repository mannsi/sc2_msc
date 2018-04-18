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

    def update(self, replay_buffer):
        """
        Update the model
        :param replay_buffer: iterable of (state, action, reward, next_state) tuples
        :return:
        """
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
        rounded_dist = int(round(dist))
        return rounded_dist
        # return str(rounded_dist)

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, val):
        self._training_mode = val


class RandomModel(Sc2Model):
    def select_action(self, obs):
        return np.random.choice(self.possible_actions)

    def update(self, replay_buffer):
        pass


class QLearningTableEnemyFocusedModel(Sc2Model):
    """ Q learning table model where location is always on SCV """
    def __init__(self, possible_actions, learning_rate, reward_decay, epsilon_greedy, total_episodes, should_decay_lr):
        super().__init__(possible_actions)
        self.possible_actions_dict = {a.internal_id: a for a in possible_actions}
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon_greedy
        self.q_table = pd.DataFrame(columns=self.possible_actions_dict.keys(), dtype=np.float64)
        self.total_episodes = total_episodes
        self.should_decay_lr = should_decay_lr
        self.episode_num = 0

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

    def update(self, replay_buffer):
        self.episode_num += 1

        if self.should_decay_lr:
            self.lr = self.lr * (self.total_episodes - self.episode_num) / self.total_episodes

        for s, a, r, s_ in replay_buffer:
            self.check_state_exist(s)
            next_state_is_final_state = s_ is None
            if not next_state_is_final_state:
                self.check_state_exist(s_)

            q_predict = self.q_table.ix[s, a.internal_id]

            if not next_state_is_final_state:
                q_target = r + self.gamma * self.q_table.ix[s_, :].max()
            else:
                q_target = r

            # update
            self.q_table.ix[s, a.internal_id] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        try:
            if state not in self.q_table.index:
                # append new state to q table.
                ser = pd.Series([0] * len(self.possible_actions), index=self.q_table.columns, name=state)
                self.q_table = self.q_table.append(ser)
        except Exception as ex:
            a = 4