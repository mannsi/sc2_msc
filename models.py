import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class Sc2Model:
    def __init__(self, actions, lr=0.1, epsilon=0.9, decay_lr=0.99, total_episodes=100):
        """
        :param actions: list of ScAction objects
        """
        self.lr = lr
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.decay_lr = decay_lr
        self._training_mode = True
        self.actions = actions
        self.episode_num = 0

    def select_action(self, obs):
        """
                Select an action given an observation
                :param obs: sc2env observation
                :return: sc_action.ScAction object
                """
        if not self.training_mode or np.random.uniform() < self.epsilon:
            return self._select_action(obs)
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def _select_action(self, obs):
        raise NotImplementedError("Please Implement this method")

    def update(self, replay_buffer):
        """
        Update the model
        :param replay_buffer: iterable of (state, action, reward, next_state) tuples
        :return:
        """
        self.episode_num += 1

        if self.decay_lr:
            self.lr = self.lr * (self.total_episodes - self.episode_num) / self.total_episodes

        return self._update(replay_buffer)

    def _update(self, replay_buffer):
        raise NotImplementedError("Please Implement this method")

    def default_action(self):
        return self.actions[0]

    def save(self, save_file):
        pass

    @staticmethod
    def load(save_file):
        pass


    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, val):
        self._training_mode = val


class RandomModel(Sc2Model):
    def __init__(self, actions):
        super().__init__(actions)

    def _select_action(self, obs):
        return np.random.choice(self.actions)

    def _update(self, replay_buffer):
        pass


class QLearningTableModel(Sc2Model):
    def __init__(self, actions, lr, reward_decay, epsilon_greedy, total_episodes, should_decay_lr):
        super().__init__(actions, lr, epsilon_greedy, should_decay_lr, total_episodes)
        self.possible_actions_dict = {a.internal_id: a for a in actions}
        self.gamma = reward_decay
        self.q_table = pd.DataFrame(columns=self.possible_actions_dict.keys(), dtype=np.float64)

    def _select_action(self, state):
        self.check_state_exist(state)

        # choose best action
        state_action_q_values = self.q_table.ix[state, :]

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

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table.
            ser = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(ser)


class Dense1DModel(Sc2Model):
    def __init__(self, actions, lr, reward_decay, epsilon_greedy, total_episodes, num_inputs, mini_batch_size, log_dir):
        super().__init__(actions, lr, epsilon_greedy, False, total_episodes)
        self.log_dir = log_dir
        self.mini_batch_size = mini_batch_size
        self.num_inputs = num_inputs
        self.gamma = reward_decay
        self.model = self.create_model()

    def _select_action(self, s):
        q_values = self.model.predict(s)
        action_index = np.argmax(q_values)
        return self.actions[action_index]

    def _update(self, replay_buffer):
        if len(replay_buffer) < self.mini_batch_size:
            return {}
        mini_batch = random.sample(replay_buffer, self.mini_batch_size)
        X_train = []
        y_train = []
        for s, a, r, s_ in mini_batch:
            y = self.model.predict(s)
            a_index = self.actions.index(a)

            if s_ is not None:
                y_next_pred = self.model.predict(s_)
                y_target = r + self.gamma * np.max(y_next_pred)
            else:
                y_target = r

            y[0][a_index] = y_target
            X_train.append(s.reshape(s.shape[1],))
            y_train.append(y.reshape(y.shape[1],))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        tb_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=5, write_graph=True, write_images=True, write_grads=True)
        # tb_callback = TensorBoard(log_dir='.Graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True)
        results = self.model.fit(X_train, y_train, batch_size=self.mini_batch_size, epochs=1, verbose=1, callbacks=[tb_callback])
        loss = results.history['loss'][0]
        return {'loss': loss}

    def save(self, save_file):
        pass

    @staticmethod
    def load(save_file):
        pass

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=150, activation='relu', input_dim=self.num_inputs,))
        model.add(Dense(units=150, activation='relu'))
        model.add(Dense(units=len(self.actions), activation='relu'))

        optimizer = Adam()
        model.compile(loss='mse', optimizer=optimizer)
        return model
