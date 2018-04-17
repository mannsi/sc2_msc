import os
import sys
import threading
import time
import random

import tensorflow as tf
import pandas as pd
from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import actions

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import Sc2Agent
from models import AlwayAttackEnemyModel, RandomModel, QLearningTableEnemyFocusedModel, MoveToEnemyThenStopModel
from sc2_action import Sc2Action
import constants

LOCK = threading.Lock()
FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("snapshot_step", 1, "Step for snapshot.")  # I use this to run the agent without training
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "/home/mannsi/code/sc2_msc/log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("minimap_resolution", 84, "Resolution for minimap feature layers.")

flags.DEFINE_string("map", "DefeatLing", "Name of a map to use.")
flags.DEFINE_integer("max_steps", 20, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
# flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_string("agent", "always_attack", "Which agent to run.")
flags.DEFINE_string("net", "atari", "atari or fcn.")

flags.DEFINE_integer("episodes_between_updates", 1, "How many episodes to run before updating agent")
flags.DEFINE_bool("randomize_replay_buffer", True, "Randomize the replay buffer before updating an agent")
flags.DEFINE_bool("test_agent", True, "To run agent both in training and test mode")
flags.DEFINE_string("run_comment", "Normal", "A comment string to distinguish the run.")

FLAGS(sys.argv)
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

replay_dir = os.path.join(run_log_path, 'Replays')
os.makedirs(replay_dir)


def run_agent(agent, map_name, visualize, tb_training_writer, tb_testing_writer):
    start_time = time.time()
    with sc2_env.SC2Env(
            map_name=map_name,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize,
            replay_dir=replay_dir,
            save_replay_episodes=FLAGS.snapshot_step) as env:
        replay_buffer = []
        for episode_number in range(1, NUM_EPISODES + 1):
            obs = env.reset()[0]  # Initial obs from env
            obses = []
            while True:
                prev_obs = obs
                action = agent.act(obs)
                obs = env.step([action.get_function_call()])[0]
                # obses.append(obs)
                s, a, r, s_ = agent.obs_to_state(prev_obs), action, obs.reward, agent.obs_to_state(obs)
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
    tb_writer.flush()


def run(unused_argv):
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    sc_actions = [
                  # Sc2Action(constants.NO_OP, actions.FUNCTIONS.no_op.id, False, False),
                  Sc2Action(constants.MOVE_TO_ENEMY, actions.FUNCTIONS.Move_screen.id, True),
                  # Sc2Action(constants.MOVE_FROM_ENEMY, actions.FUNCTIONS.Move_screen.id, True),
                  # Sc2Action(constants.ATTACK_ENEMY, actions.FUNCTIONS.Attack_screen.id, True)
                  ]

    if FLAGS.agent == "always_attack":
        model = AlwayAttackEnemyModel([Sc2Action(constants.ATTACK_ENEMY, actions.FUNCTIONS.Attack_screen.id, True)])
        # model = AlwayAttackEnemyModel([Sc2Action(constants.MOVE_TO_ENEMY, actions.FUNCTIONS.Move_screen.id, True)])
    elif FLAGS.agent == "move_then_stop":
        sc_actions = [
            Sc2Action(constants.MOVE_TO_ENEMY, actions.FUNCTIONS.Move_screen.id, True),
            Sc2Action(constants.NO_OP, actions.FUNCTIONS.no_op.id, False, False),
        ]
        model = MoveToEnemyThenStopModel(sc_actions)
    elif FLAGS.agent == "random":
        model = RandomModel(sc_actions)
    elif FLAGS.agent == "table":
        model = QLearningTableEnemyFocusedModel(possible_actions=sc_actions,
                                                learning_rate=FLAGS.learning_rate,
                                                reward_decay=FLAGS.discount,
                                                epsilon_greedy=0.9)
    else:
        raise NotImplementedError()

    agent = Sc2Agent(model)

    tb_training_writer = tf.summary.FileWriter(TRAIN_LOG)
    tb_testing_writer = tf.summary.FileWriter(TEST_LOG)

    run_agent(agent, FLAGS.map, FLAGS.render, tb_training_writer, tb_testing_writer)

    tb_training_writer.close()
    tb_testing_writer.close()


if __name__ == "__main__":
    # Run the agent
    app.run(run)
