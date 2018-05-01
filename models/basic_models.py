import os
import numpy as np
import pandas as pd
from .sc2_model import Sc2Model
import constants


class RandomModel(Sc2Model):
    def __init__(self, actions):
        super().__init__(actions)
        self.epsilon = 0  # Zero chance the model has to make a choice

    def _select_action(self, obs, illegal_internal_action_ids=None):
        pass

    def _update(self, replay_buffer):
        pass


class PredefinedActionsModel(Sc2Model):
    def __init__(self, actions, list_of_internal_action_ids):
        super().__init__(actions)
        self.list_of_internal_action_ids = list_of_internal_action_ids
        self.epsilon = 1  # Zero chance of random choice

    def _select_action(self, state, illegal_internal_action_ids=None):
        internal_action_id = self.list_of_internal_action_ids.pop(0)
        for action in self.actions:
            if action.internal_id == internal_action_id and internal_action_id not in illegal_internal_action_ids:
                return action
        return super().default_action()

    def _update(self, replay_buffer):
        pass


class CmdInputModel(Sc2Model):
    def __init__(self, actions, key_to_action_mapping):
        super().__init__(actions)
        self.key_to_action_mapping = key_to_action_mapping
        self.epsilon = 1  # Zero chance of random choice

        print(self.key_to_action_mapping)

    def _select_action(self, obs, illegal_internal_action_ids=None):
        keyboard_input = input()
        try:
            internal_action_id = self.key_to_action_mapping[keyboard_input]
            while internal_action_id in illegal_internal_action_ids:
                print(f'Illegal action {internal_action_id}. Try another one')
                internal_action_id = self.key_to_action_mapping[input()]
        except KeyError:
            internal_action_id = "_NO_OP"

        for action in self.actions:
            if action.internal_id == internal_action_id:
                return action
        raise ValueError(f"Internal action id {internal_action_id} not found in possible actions")

    def _update(self, replay_buffer):
        pass


class HardCodedTableAgent(Sc2Model):
    def __init__(self, actions):
        super().__init__(actions)
        self.epsilon = 1  # Zero chance of random choice

    def _select_action(self, obs, illegal_internal_action_ids=None):
        distance, flying, enemmy_coming = self._state_to_index_vars(obs)

        if flying == 1:
            internal_action_id = constants.LAND
        else:
            if distance == 0:
                if enemmy_coming == 0:
                    internal_action_id = constants.ATTACK_ENEMY
                else:
                    internal_action_id = constants.FLIGHT
            elif distance == 1:
                if enemmy_coming == 0:
                    internal_action_id = constants.ATTACK_ENEMY
                else:
                    internal_action_id = constants.FLIGHT
            elif distance == 2:
                internal_action_id = constants.ATTACK_ENEMY
            else:
                raise ValueError(f"Unknown distance mapping {distance}")
        for action in self.actions:
            if action.internal_id == internal_action_id and internal_action_id not in illegal_internal_action_ids:
                return action
        return super().default_action()

    def _update(self, replay_buffer):
        pass

    @staticmethod
    def _state_to_index_vars(state):
        distance = int(state[0:1])
        flying = int(state[1:2])
        enemy_coming = int(state[2:3])
        return distance, flying, enemy_coming


class QLearningTableModel(Sc2Model):
    def __init__(self, actions, lr, reward_decay, epsilon_greedy, total_episodes, decay_lr, decay_epsilon):
        super().__init__(actions, lr, epsilon_greedy, decay_lr, decay_epsilon, total_episodes)
        self.possible_actions_dict = {a.internal_id: a for a in actions}
        self.gamma = reward_decay
        self.q_table = pd.DataFrame(columns=self.possible_actions_dict.keys(), dtype=np.float64)

    def _select_action(self, state, illegal_internal_action_ids=None):
        self.check_state_exist(state)

        if illegal_internal_action_ids is None:
            illegal_internal_action_ids = []

        # choose best action
        state_action_q_values = self.q_table.loc[state, self.q_table.columns.difference(illegal_internal_action_ids)]

        # Shuffle to select randomly if some states are equal
        state_action_q_values = state_action_q_values.reindex(np.random.permutation(state_action_q_values.index))

        best_internal_action_id_for_state = state_action_q_values.idxmax()
        return self.possible_actions_dict[best_internal_action_id_for_state]

    def _update(self, replay_buffer):
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
        return {}

    def save(self, save_folder):
        self.q_table.to_csv(os.path.join(save_folder, 'model.csv'))

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table.
            ser = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(ser)
