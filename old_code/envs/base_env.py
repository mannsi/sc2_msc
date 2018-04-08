import logging

import gym
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

from old_code.utils import sc2_log

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]


class BaseEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = dict(
        feature_screen_size=84,
        feature_minimap_size=64
    )

    def __init__(self, **kwargs):
        super().__init__()
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
        # Returns possible actions as a gym.Space object
        raise NotImplementedError()

    def _translate_action(self, action_params):
        # List of action ints to sc2 actions
        raise NotImplementedError()

    def _get_observation_space(self):
        # Get observation space which is a 2D map with player id int values. Returns a gym.Space object
        raise NotImplementedError()

    def _extract_observation(self, obs):
        # Take raw observations, filter them down to just player_relative feature map and finally reshape.
        raise NotImplementedError()

    def reset(self):
        if self._episode > 0:
            sc2_log.to_file(logging.INFO,
                           f"Episode {self._episode} ended with reward {self._episode_reward} after {self._num_step} steps.")
            sc2_log.to_file(logging.INFO, f"Got {self._total_reward} total reward so far, with an average reward of {float(self._total_reward) / self._episode} per episode")
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        sc2_log.to_file(logging.INFO, f"Episode {self._episode} starting...", )
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
            sc2_log.to_file(logging.INFO, "Interrupted. Quitting...")
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

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        sc2_log.to_file(logging.DEBUG, f"Initializing SC2Env with settings: {args}")
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

    def render(self, mode='human'):
        pass