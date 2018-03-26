import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import sc2gym
from absl import flags

FLAGS = flags.FLAGS
FLAGS([__file__])

# Create an OpenAIgym environment
# ReversedAddition-v0
# CartPole-v0
env = OpenAIGym('SC2CollectMineralShards-v2', visualize=False)

print(env.states)
print(env.actions)

# Network as list of layers
network_spec = [
    dict(type='conv2d', size=64),
    dict(type='flatten'),
    dict(type='dense', size=32, activation='relu'),
    dict(type='lstm', size=64)
]

saver_spec = {
    'directory': './model',
    'seconds': 3600
}

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    # batch_size=10,
    # Agent
    # preprocessing=None,
    # exploration=None,
    reward_preprocessing=None,
    saver=saver_spec,
    # BatchAgent
    # keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-5
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode="states",
    baseline={
        "type": "cnn",
        "conv_sizes": [32],
        "dense_sizes": [32]
    },
    baseline_optimizer={
        "type": "multi_step",
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-5
        },
        "num_steps": 10
    },
    gae_lambda=0.99,
    # normalize_rewards=False,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # summary=None,
    # distributed=None
)

print('partially success')

# Create the runner
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
rewards = []


def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    global rewards
    rewards += [r.episode_rewards[-1]]
    return True


# Start learning
runner.run(episodes=60000, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)