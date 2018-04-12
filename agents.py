from pysc2.lib import features, actions
from sc2_env_functions import get_own_unit_location, get_enemy_unit_location

from sc_action import ScAction
import constants


class Sc2Agent:
    def __init__(self, model):
        """
        :param model: Sc2Model object
        """
        self.model = model

    def act(self, obs):
        """
        Take a step and return an action using the policy
        :param obs: SC2Env state
        :return: SC2Action
        """
        if not self.marine_selected(obs):
            return ScAction(actions.FUNCTIONS.no_op.id, has_location=False).get_function_call()
        return self._act(obs)

    def _act(self, obs):
        sc_action = self.model.select_action(obs)

        if sc_action.internal_id == constants.NO_OP:
            action = sc_action.get_function_call()
        elif sc_action.internal_id == constants.ATTACK_ENEMY:
            location = get_enemy_unit_location(obs)
            action = sc_action.get_function_call(location)
        elif sc_action.internal_id == constants.MOVE_TO_ENEMY:
            location = get_enemy_unit_location(obs)
            action = sc_action.get_function_call(location)
        elif sc_action.internal_id == constants.MOVE_FROM_ENEMY:
            pass
        else:
            raise NotImplementedError("Unknown action ID received")

        return action

    def observe(self, replay_buffer):
        """
        Update the agent
        :param replay_buffer: list of tuples containing (state, action, reward, next_state)
        """
        for s, a, r, s_ in replay_buffer:
            self.model.update(s, a, r, s_)

    def obs_to_state(self, obs):
        return self.model.obs_to_state(obs)

    @property
    def training_mode(self):
        return self.model.training_mode

    @training_mode.setter
    def training_mode(self, val):
        self.model.training_mode = val

    @staticmethod
    def marine_selected(obs):
        # For some reason the environment looses selection of my marine
        return actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']
