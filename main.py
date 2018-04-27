import os
import time

import tensorflow as tf
from absl import app

from pysc2.env import sc2_env
from pysc2.lib import stopwatch

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import Sc2Agent, Simple1DAgent, SimpleVikingAgent
from models.basic_models import RandomModel, QLearningTableModel, PredefinedActionsModel, CmdInputModel, HardCodedTableAgent

from sc2_action import Sc2Action
import constants
import flags_import

import random
import numpy as np

random.seed(7)
np.random.seed(7)


FLAGS = flags_import.get_flags()

if FLAGS.should_log:
    BASE_LOG_PATH = os.path.join(FLAGS.log_path, FLAGS.agent, FLAGS.map, str(FLAGS.step_mul))

    if FLAGS.run_comment is not "":
        BASE_LOG_PATH = os.path.join(BASE_LOG_PATH, FLAGS.run_comment)

    # Have incremental log counter runs
    log_counter = 0
    while True:
        log_counter += 1
        run_log_path = os.path.join(BASE_LOG_PATH, str(log_counter))
        if not os.path.exists(run_log_path):
            TRAIN_LOG = os.path.join(run_log_path, 'TRAIN')
            os.makedirs(TRAIN_LOG)

            if FLAGS.test_agent:
                TEST_LOG = os.path.join(run_log_path, 'TEST')
                os.makedirs(TEST_LOG)
            break

    if FLAGS.save_replays:
        replay_dir = os.path.join(run_log_path, 'Replays')
        os.makedirs(replay_dir)
        save_replays_every_num_episodes = FLAGS.snapshot_step
    else:
        replay_dir = None
        save_replays_every_num_episodes = 0

    agent_save_files_dir = os.path.join(run_log_path, 'AgentFiles')
    os.makedirs(agent_save_files_dir)
else:
    replay_dir = None
    save_replays_every_num_episodes = 0


def run_agent(agent, map_name, visualize, tb_training_writer, tb_testing_writer):
    start_time = time.time()
    with sc2_env.SC2Env(
            map_name=map_name,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize,
            replay_dir=replay_dir,
            save_replay_episodes=save_replays_every_num_episodes) as env:
        replay_buffer = []
        step_counter = 0
        for episode_number in range(1, FLAGS.max_steps + 1):
            if not agent.training_mode:
                print("Next episode is test episode")
            obs = env.reset()[0]  # Initial obs from env
            while True:
                step_counter += 1
                prev_obs = obs
                action = agent.act(obs)
                obs = env.step([action.get_function_call()])[0]
                s = agent.obs_to_state(prev_obs)
                a = action
                r = obs.reward
                s_ = agent.obs_to_state(obs) if not obs.last() else None

                # print(f'Action {a.internal_id}, next state {agent.obs_to_state(obs)}, '
                #       f'next legal actions {agent.get_legal_internal_action_ids(obs)}')

                replay_buffer.append((s, a, r, s_))

                # if step_counter <= FLAGS.experience_replay_max_size:
                #     # Start by filling the buffer
                #     replay_buffer.append((s, a, r, s_))
                # else:
                #     # Now systematically replace the oldest values
                #     replay_buffer[step_counter % FLAGS.experience_replay_max_size] = (s, a, r, s_)

                if obs.last() or r < 0:
                    import pandas as pd
                    replay_buffer_df = pd.DataFrame.from_records(replay_buffer,
                                                                 columns=['state', 'action', 'reward', 'next_state'])

                    print(f'Episode rew: {obs.observation["score_cumulative"][0]}')

                    if agent.training_mode:
                        agent.observe(replay_buffer)
                        log_episode(tb_training_writer, obs, episode_number, None)
                    else:
                        log_episode(tb_testing_writer, obs, episode_number, None)
                        if FLAGS.should_log:
                            agent.save(os.path.join(agent_save_files_dir, str(episode_number)))

                    replay_buffer = []

                    should_test_agent = FLAGS.test_agent and episode_number % FLAGS.snapshot_step == 0
                    if should_test_agent:
                        # Next episode will be a test episode
                        agent.training_mode = False
                    else:
                        agent.training_mode = True

                    break  # Exit episode
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


def log_episode(tb_writer, last_obs, episode_number, agent_results_dict):
    if tb_writer is None:
        return
    total_episode_rewards = last_obs.observation["score_cumulative"][0]
    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode rewards', simple_value=total_episode_rewards)])
    tb_writer.add_summary(reward_summary, episode_number)

    if agent_results_dict is not None:
        for k, v in agent_results_dict.items():
            results_summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
            tb_writer.add_summary(results_summary, episode_number)


def create_agent(sc_actions):
    """
    Creates an sc2 agent based on available actions and Flags variables
    :param sc_actions: List of Sc2Action objects that define which actions are available to the agent
    :return: agents.SC2Agent object
    """
    if FLAGS.agent == "always_attack":
        model = RandomModel([Sc2Action(constants.ATTACK_ENEMY)])
        agent = Sc2Agent(model)
    elif FLAGS.agent == "random":
        model = RandomModel(sc_actions)
        agent = SimpleVikingAgent(model, sc_actions)
    elif FLAGS.agent == "predefined_actions":
        action_list = [constants.ATTACK_ENEMY, constants.ATTACK_ENEMY, constants.FLIGHT, constants.LAND, constants.ATTACK_ENEMY, constants.ATTACK_ENEMY] * 100
        model = PredefinedActionsModel(sc_actions, action_list)
        agent = SimpleVikingAgent(model, sc_actions)
    elif FLAGS.agent == "hardcoded":
        model = HardCodedTableAgent(sc_actions)
        agent = SimpleVikingAgent(model, sc_actions)
    elif FLAGS.agent == "cmd_input":
        key_to_action_mapping = {'f': constants.FLIGHT,
                                 'l': constants.LAND,
                                 'n': constants.NO_OP,
                                 'a': constants.ATTACK_ENEMY,
                                 'm': constants.MOVE_FROM_ENEMY}
        model = CmdInputModel(sc_actions, key_to_action_mapping)
        agent = SimpleVikingAgent(model, sc_actions)
    elif FLAGS.agent == "table":
        model = QLearningTableModel(actions=sc_actions,
                                    lr=FLAGS.learning_rate,
                                    reward_decay=FLAGS.discount,
                                    epsilon_greedy=0.9,
                                    total_episodes=FLAGS.max_steps,
                                    should_decay_lr=FLAGS.decay_lr)
        agent = SimpleVikingAgent(model, sc_actions)
    elif FLAGS.agent == "1d_qlearning":
        num_inputs = FLAGS.screen_resolution * FLAGS.screen_resolution * 2  # 2 for our and enemy unit plains
        from models.dl_models import Dense1DModel  # Import here because importing TensorFlow is slow
        model = Dense1DModel(actions=sc_actions,
                             lr=FLAGS.learning_rate,
                             reward_decay=FLAGS.discount,
                             epsilon_greedy=0.9,
                             total_episodes=FLAGS.max_steps,
                             num_inputs=num_inputs,
                             mini_batch_size=FLAGS.mini_batch_size,
                             log_dir=os.path.join(TRAIN_LOG, 'model'))
        agent = Simple1DAgent(model)
    else:
        raise NotImplementedError()
    return agent


def run(unused_argv):
    print(unused_argv)
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    sc_actions = [
        Sc2Action(constants.NO_OP),
        # Sc2Action(constants.MOVE_TO_ENEMY),
        # Sc2Action(constants.MOVE_FROM_ENEMY),
        Sc2Action(constants.ATTACK_ENEMY),
        Sc2Action(constants.LAND),
        Sc2Action(constants.FLIGHT),
    ]

    agent = create_agent(sc_actions)

    tb_training_writer = None
    tb_testing_writer = None

    if FLAGS.should_log:
        tb_training_writer = tf.summary.FileWriter(TRAIN_LOG)
        if FLAGS.test_agent:
            tb_testing_writer = tf.summary.FileWriter(TEST_LOG)

    run_agent(agent, FLAGS.map, FLAGS.render, tb_training_writer, tb_testing_writer)


if __name__ == "__main__":
    # Run the agent
    app.run(run)
