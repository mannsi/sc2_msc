# noinspection PyUnresolvedReferences
import envs

# noinspection PyUnresolvedReferences
import maps

import argparse
import numpy as np
import gym
from absl import flags
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import agents.ppo
import networks.first_network


__description__ = 'Run a scripted example using the SC2MoveToBeacon-v1 environment.'
_PLAYER_NEUTRAL = 3  # beacon/minerals


class MoveToBeacon2d:
    def __init__(self, step_mul=None):
        self.env_name = "MarineVsScvEnv-v0"
        self.step_mul = step_mul

    def run(self, num_episodes=1):
        environment = OpenAIGym(self.env_name)

        print(environment.states)
        print(environment.actions)

        env = gym.make(self.env_name)
        env.settings['step_mul'] = self.step_mul

        episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = env.reset()

            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, reward, done, _ = env.step(action)

            episode_rewards[ix] = env.episode_reward

        env.close()

        return episode_rewards

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        if not neutral_y.any():
            # raise Exception("Beacon not found!")
            target = [0, 0]
        else:
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
        return target


rewards = []

def main():
    FLAGS = flags.FLAGS
    FLAGS([__file__])

    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='number of episodes to run')
    parser.add_argument('--step-mul', type=int, default=None,
                        help='number of game steps to take per turn')
    parser.add_argument('--agent_type', type=str, default='ppo',
                        help='Which of the predefined agents to run')
    parser.add_argument('--env_id', type=str, default='MarineVsScvEnv-v0',
                        help='Id of the environment to use. See envs package for possible envs')
    parser.add_argument('--network', type=str, default='first_network',
                        help='Which network configuration to use')
    args = parser.parse_args()

    env = OpenAIGym(args.env_id)

    # saver = {
    #     'directory': './model',
    #     'seconds': 3600
    # }
    saver = None

    if args.network == 'first_network':
        network = networks.first_network.get_network()
    else:
        network = networks.first_network.get_network()

    if args.agent_type == 'ppo':
        agent = agents.ppo.get_agent(env, network, saver)
    else:
        agent = agents.ppo.get_agent(env, network, saver)

    # Create the runner
    runner = Runner(agent=agent, environment=env)

    def episode_finished(r):
        print(
            "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1]))
        global rewards
        rewards += [r.episode_rewards[-1]]
        return True

    runner.run(episodes=10, episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )


if __name__ == "__main__":
    main()
