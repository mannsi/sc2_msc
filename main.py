import os
import time
import random

import tensorflow as tf
import pandas as pd
import numpy as np
from absl import app

from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import actions

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import Sc2Agent
from models import RandomModel, QLearningTableEnemyFocusedModel
from sc2_action import Sc2Action
import constants
import flags_import

FLAGS = flags_import.get_flags()

np.seterr(all='raise')

random.seed(7)
np.random.seed(7)

BASE_LOG_PATH = os.path.join(FLAGS.log_path, FLAGS.agent, FLAGS.map, str(FLAGS.step_mul))

if FLAGS.run_comment is not "":
    BASE_LOG_PATH = os.path.join(BASE_LOG_PATH, FLAGS.run_comment)

NUM_EPISODES = FLAGS.max_steps

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
        for episode_number in range(1, NUM_EPISODES + 1):
            obs = env.reset()[0]  # Initial obs from env
            while True:
                prev_obs = obs
                action = agent.act(obs)
                obs = env.step([action.get_function_call()])[0]
                s, a, r, s_ = agent.obs_to_state(prev_obs), action, obs.reward, agent.obs_to_state(obs) if not obs.last() else None
                replay_buffer.append((s, a, r, s_))

                if obs.last():
                    replay_buffer_df = pd.DataFrame.from_records(replay_buffer, columns=['state', 'action', 'reward', 'next_state'])

                    should_update_agent = episode_number % FLAGS.episodes_between_updates == 0
                    if should_update_agent:
                        if FLAGS.randomize_replay_buffer:
                            random.shuffle(replay_buffer)
                        agent.observe(replay_buffer)
                        replay_buffer = []

                    if agent.training_mode:
                        log_episode(tb_training_writer, obs, episode_number)
                    else:
                        log_episode(tb_testing_writer, obs, episode_number)
                        agent.save(os.path.join(agent_save_files_dir, str(episode_number)))

                    should_test_agent = FLAGS.test_agent and episode_number % FLAGS.snapshot_step == 0
                    if should_test_agent:
                        # Next episode will be a test episode
                        agent.training_mode = False
                    else:
                        agent.training_mode = True

                    break  # Exit episode
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


def log_episode(tb_writer, last_obs, episode_number):
    total_episode_rewards = last_obs.observation["score_cumulative"][0]
    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode rewards', simple_value=total_episode_rewards)])
    tb_writer.add_summary(reward_summary, episode_number)


def run(unused_argv):
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    sc_actions = [
                  Sc2Action(constants.NO_OP, actions.FUNCTIONS.no_op.id, False, False),
                  Sc2Action(constants.MOVE_TO_ENEMY, actions.FUNCTIONS.Move_screen.id, True),
                  Sc2Action(constants.MOVE_FROM_ENEMY, actions.FUNCTIONS.Move_screen.id, True),
                  Sc2Action(constants.ATTACK_ENEMY, actions.FUNCTIONS.Attack_screen.id, True)
                  ]

    if FLAGS.agent == "always_attack":
        model = RandomModel([Sc2Action(constants.ATTACK_ENEMY, actions.FUNCTIONS.Attack_screen.id, True)])
        # model = AlwayAttackEnemyModel([Sc2Action(constants.MOVE_TO_ENEMY, actions.FUNCTIONS.Move_screen.id, True)])
    elif FLAGS.agent == "random":
        model = RandomModel(sc_actions)
    elif FLAGS.agent == "table":
        model = QLearningTableEnemyFocusedModel(possible_actions=sc_actions,
                                                learning_rate=FLAGS.learning_rate,
                                                reward_decay=FLAGS.discount,
                                                epsilon_greedy=0.9,
                                                total_episodes=NUM_EPISODES,
                                                should_decay_lr=True)
    else:
        raise NotImplementedError()

    agent = Sc2Agent(model)

    tb_training_writer = tf.summary.FileWriter(TRAIN_LOG)
    if FLAGS.test_agent:
        tb_testing_writer = tf.summary.FileWriter(TEST_LOG)
    else:
        tb_testing_writer = None

    run_agent(agent, FLAGS.map, FLAGS.render, tb_training_writer, tb_testing_writer)


if __name__ == "__main__":
    # Run the agent
    app.run(run)
