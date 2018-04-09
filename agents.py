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

        return SimpleState(dx, dy)

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
        super().__init__()
        self.discount = discount
        self.step_size = step_size
        self.num_possible_actions = 3
        self.q_table = np.zeros(shape=(x_size, y_size, self.num_possible_actions))
        self.epsilon = 0.1

    def step(self, obs):
        if not self.marine_selected(obs):
            return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])

        do_random_action = np.random.rand() < self.epsilon
        if do_random_action and self.is_training:
            action_index = np.random.randint(self.num_possible_actions)
        else:
            state = self.sc2obs_to_table_state(obs)
            action_index = self.get_max_action_val_index(state)

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
        # Update formulas
        # delta = (r + self.discount * max(Q(s_n+1,:))) - Q(s_n,a_n)
        # Q(s_n,a_n) = Q(s_n,a_n) + self.step_size * delta

        for obs, action, reward, next_obs in replay_buffer:
            if action.function == actions.FUNCTIONS.no_op.id:
                internal_function_action = TableAgent.ACTION_NO_OP
            elif action.function == actions.FUNCTIONS.Move_screen.id:
                internal_function_action = TableAgent.ACTION_MOVE_TO_ENEMY
            elif action.function == actions.FUNCTIONS.Attack_screen.id:
                internal_function_action = TableAgent.ACTION_ATTACK_ENEMY
            elif action.function == actions.FUNCTIONS.select_army.id:
                # Not an actual action, just because of a glitch in the env
                continue
            else:
                raise ValueError(f"Received an unexepected action function id of value {action.function}")

            # Get current Q val
            current_state = self.sc2obs_to_table_state(obs)
            q_current = self.q_table[current_state.dx, current_state.dy, internal_function_action]

            # Get highest next Q val
            next_state = self.sc2obs_to_table_state(next_obs)
            index_for_max_action_value_next = self.get_max_action_val_index(next_state)
            q_next_max = self.q_table[next_state.dx, next_state.dy, index_for_max_action_value_next]

            # Update current Q val using the Bellman equation
            delta = (reward + self.discount * q_next_max) - q_current
            self.q_table[current_state.dx, current_state.dy, internal_function_action] = q_current + self.step_size * delta

    def get_max_action_val_index(self, state):
        q_values = self.q_table[state.dx, state.dy, :]
        indexes_with_max_val = np.flatnonzero(q_values == q_values.max())
        action_index = np.random.choice(indexes_with_max_val)
        return action_index


class SimpleState:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
