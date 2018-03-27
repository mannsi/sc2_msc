import logging
import numpy as np
import gym
from gym import spaces
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features


_MAP_NAME = 'DefeatScv'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

# Available actions
_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_INDEX = 0
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_SCREEN_INDEX = 1
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_SCREEN_INDEX = 2

# Not a real action but performed at start of episode
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MarineVsScvEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = dict(
        feature_screen_size=84,
        feature_minimap_size=64,
        visualize=False,
    )

    def __init__(self, **kwargs):
        super().__init__()
        kwargs['map_name'] = _MAP_NAME
        self._kwargs = kwargs
        self._env = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

        self.available_actions = None
        self._action_space = None
        self._observation_space = None

    def _get_action_space(self):
        # Returns possible actions.
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        num_available_actions = 3
        return spaces.MultiDiscrete([num_available_actions] + [s for s in screen_shape])

    def _translate_action(self, action_params):
        # Maps the action parameter (list of ints) to an actual agent action
        action_index = action_params[0]

        target = action_params[1:]
        if action_index == _NO_OP_INDEX:
            return [_NO_OP]
        elif action_index == _MOVE_SCREEN_INDEX:
            return [_MOVE_SCREEN, _NOT_QUEUED, target]
        elif action_index == _ATTACK_SCREEN_INDEX:
            return [_ATTACK_SCREEN, _NOT_QUEUED, target]
        else:
            raise ValueError(f"Got unexpected action index {action_index}")

    def reset(self):
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward so far, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        logger.info("Episode %d starting...", self._episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._extract_observation(obs)
        return obs

    def step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._safe_step(action)
        if obs is None:
            return None, 0, True, {}
        obs = self._extract_observation(obs)
        return obs, reward, done, info

    def _safe_step(self, action):
        self._num_step += 1
        try:
            obs = self._env.step([actions.FunctionCall(action[0], action[1:])])[0]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}
        # except Exception:
        #     logger.exception("An unexpected error occurred while applying action to environment.")
        #     return None, 0, True, {}
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        is_last_step = obs.step_type == StepType.LAST
        return obs, reward, is_last_step, {}

    def _get_observation_space(self):
        # TODO edit so more feature maps are available to the agent.
        # Get observation space which is a 2D map with player id int values
        screen_shape = (1, ) + self.observation_spec[0]["feature_screen"][1:]  # Convert from 17 feature maps to 1
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape)
        return space

    def _extract_observation(self, obs):
        # TODO I probably need to edit this so the agent sees f.x. unit health. Maybe I also need to edit so agent sees multiple timesteps ?
        # Take raw observations, filter them down to just player_relative feature map and finally reshape.
        obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        obs = obs.reshape(self.observation_space.shape)
        return obs

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        logger.debug("Initializing SC2Env with settings: %s", args)
        self._env = sc2_env.SC2Env(**args)

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def close(self):
        self._env.close()
        super().close()

    @property
    def action_space(self):
        # Get possible actions that an agent can take
        if self._action_space is None:
            self._action_space = self._get_action_space()
        return self._action_space

    @property
    def observation_space(self):
        # Get observation space which is a 2D map with player id int values
        if self._observation_space is None:
            self._observation_space = self._get_observation_space()
        return self._observation_space

    @property
    def observation_spec(self):
        # Return possible observations
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def settings(self):
        return self._kwargs

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward
