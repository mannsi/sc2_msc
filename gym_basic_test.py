import gym
import sc2gym.envs

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

environment = OpenAIGym('SC2CollectMineralShards-v2', visualize=False)
# env = gym.make('SC2CollectMineralShards-v2')
# print(env.states)
# print(env.actions)

print(environment.states)
print(environment.actions)

