import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from .sc2_model import Sc2Model


class Dense1DModel(Sc2Model):
    def __init__(self, actions, lr, reward_decay, epsilon_greedy, total_episodes, num_inputs, mini_batch_size, log_dir):
        super().__init__(actions, lr, epsilon_greedy, False, total_episodes)
        self.log_dir = log_dir
        self.mini_batch_size = mini_batch_size
        self.num_inputs = num_inputs
        self.gamma = reward_decay
        self.model = self._create_model()

    def _select_action(self, s, illegal_internal_action_ids=None):
        # TODO remove illegal options
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
        tb_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=5,
                                  write_graph=True, write_images=True, write_grads=True)
        results = self.model.fit(X_train, y_train, batch_size=self.mini_batch_size, epochs=1,
                                 verbose=1, callbacks=[tb_callback])
        loss = results.history['loss'][0]
        return {'loss': loss}

    def save(self, save_file):
        pass

    @staticmethod
    def load(save_file):
        pass

    def _create_model(self):
        model = Sequential()
        model.add(Dense(units=150, activation='relu', input_dim=self.num_inputs,))
        model.add(Dense(units=150, activation='relu'))
        model.add(Dense(units=len(self.actions), activation='relu'))

        optimizer = Adam()
        model.compile(loss='mse', optimizer=optimizer)
        return model
