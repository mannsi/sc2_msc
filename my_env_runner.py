# noinspection PyUnresolvedReferences
import envs

# noinspection PyUnresolvedReferences
import maps

import argparse
import numpy as np
from absl import flags
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from agents import ppo, random, always_attack_scv, deep_q
from networks import conv_net, lstm_net

import logging
import my_log

rewards = []


def main():
    FLAGS = flags.FLAGS
    FLAGS([__file__])

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='number of episodes to run')
    parser.add_argument('--step_mul', type=int, default=8,
                        help='number of game steps to take per turn')
    parser.add_argument('--agent_type', type=str, default='always_attack_scv',
                        help='Which of the predefined agents to run')
    parser.add_argument('--env_id', type=str, default='MarineVsScvAttackOnly-v0',
                        help='Id of the environment to use. See envs package for possible envs')
    parser.add_argument('--network', type=str, default='conv',
                        help='Which network configuration to use')
    parser.add_argument('--render', action='store_true', help='To render or not using pygame. Default off')
    parser.add_argument('--map_name', type=str, default='DefeatScv',
                        help='Map to use')
    args = parser.parse_args()

    env = OpenAIGym(args.env_id)
    env.gym.default_settings['step_mul'] = args.step_mul
    env.gym.default_settings['map_name'] = args.map_name
    env.gym.default_settings['visualize'] = args.render

    # saver = {
    #     'directory': './model',
    #     'seconds': 3600
    # }
    saver = None
    my_log.init_file_logging(logging.INFO, 'output.txt')

    if args.network == 'conv':
        network = conv_net.get_network()
    elif args.network == 'lst':
        network = lstm_net.get_network()
    else:
        network = conv_net.get_network()

    if args.agent_type == 'ppo':
        agent = ppo.get_agent(env, network, saver)
    elif args.agent_type == 'deep_q':
        agent = deep_q.get_agent(env, network, saver)
    elif args.agent_type == 'random':
        agent = random.get_agent(env, saver)
    elif args.agent_type == 'always_attack_scv':
        agent = always_attack_scv.get_agent(env, saver)
    else:
        agent = ppo.get_agent(env, network, saver)

    # Create the runner
    runner = Runner(agent=agent, environment=env)

    def episode_finished(r):
        print(
            "Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                   reward=r.episode_rewards[-1]))
        global rewards
        rewards += [r.episode_rewards[-1]]
        return True

    runner.run(episodes=args.num_episodes, episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=runner.episode,
        ar=np.mean(runner.episode_rewards[-100:]))
    )


if __name__ == "__main__":
    main()
