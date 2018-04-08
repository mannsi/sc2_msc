from gym import spaces
from pysc2.lib import actions, features

from old_code.envs.base_env import BaseEnv

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

# Available actions
_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_ACTION_INDEX = 0
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_SCREEN_ACTION_INDEX = 1
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_SCREEN_ACTION_INDEX = 2

_NOT_QUEUED = [0]


class AttackOnly(BaseEnv):
    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        num_available_actions = 1
        return spaces.MultiDiscrete([num_available_actions] + [s for s in screen_shape])

    def _translate_action(self, action_params):
        target = action_params[1:]

        if _ATTACK_SCREEN not in self.available_actions:
            return [_NO_OP]
        return [_ATTACK_SCREEN, _NOT_QUEUED, target]

    def _get_observation_space(self):
        screen_shape = (1, ) + self.observation_spec[0]["feature_screen"][1:]  # Convert from 17 feature maps to 1
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape)
        return space

    def _extract_observation(self, obs):
        obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        obs = obs.reshape(self.observation_space.shape)
        return obs


class AttackMoveNoop(BaseEnv):
    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        num_available_actions = 3  # Attack, Move, NoOp
        return spaces.MultiDiscrete([num_available_actions] + [s for s in screen_shape])

    def _translate_action(self, action_params):
        action_index = action_params[0]
        target = action_params[1:]

        if action_index == _NO_OP_ACTION_INDEX:
            return [_NO_OP]
        elif action_index == _MOVE_SCREEN_ACTION_INDEX:
            if _MOVE_SCREEN not in self.available_actions:
                return [_NO_OP]
            return [_MOVE_SCREEN, _NOT_QUEUED, target]
        elif action_index == _ATTACK_SCREEN_ACTION_INDEX:
            if _ATTACK_SCREEN not in self.available_actions:
                return [_NO_OP]
            return [_ATTACK_SCREEN, _NOT_QUEUED, target]
        else:
            raise ValueError(f"Got unexpected action index {action_index}")

    def _get_observation_space(self):
        # TODO edit so more feature maps are available to the agent. Probably needs to go from Box to MultiDiscrete
        screen_shape = (1, ) + self.observation_spec[0]["feature_screen"][1:]  # Convert from 17 feature maps to 1
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape)
        return space

    def _extract_observation(self, obs):
        # TODO I probably need to edit this so the agent sees f.x. unit health. Maybe I also need to edit so agent sees multiple timesteps ?
        # TODO Then probably needs to go from Box to MultiDiscrete
        obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        obs = obs.reshape(self.observation_space.shape)
        return obs