import numpy as np


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

    def select_action(self, obs, illegal_internal_action_ids=None):
        """
                Select an action given an observation
                :param obs: sc2env observation
                :param illegal_internal_action_ids: List of illegal action ids
                :return: sc_action.ScAction object
                """
        if not self.training_mode or np.random.uniform() < self.epsilon:
            return self._select_action(obs, illegal_internal_action_ids)
        else:
            # choose random action
            if illegal_internal_action_ids is None:
                illegal_internal_action_ids = []
            action = np.random.choice([a for a in self.actions if a.internal_id not in illegal_internal_action_ids])

        return action

    def _select_action(self, obs, illegal_internal_action_ids=None):
        raise NotImplementedError("Please Implement this method")

    def update(self, replay_buffer):
        """
        Update the model
        :param replay_buffer: iterable of (state, action, reward, next_state) tuples
        :return:
        """
        if self.decay_lr:
            self.lr -= 1/self.total_episodes

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
