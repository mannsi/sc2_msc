import os
import sys
import threading
import time
import random

import tensorflow as tf
from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import actions

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import TableAgent, AlwaysAttackAgent, RandomAgent
from sc_action import ScAction

LOCK = threading.Lock()
FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("snapshot_step", int(1e2), "Step for snapshot.")  # I use this to run the agent without training
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_string("map", "DefeatScv", "Name of a map to use.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_integer("minimap_resolution", 84, "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_steps", 10, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("max_agent_steps", 100,
                     "Total agent steps.")  # Max agent steps per episode. Runs these steps or until it finishes
flags.DEFINE_string("agent", "always_attack", "Which agent to run.")
flags.DEFINE_string("net", "atari", "atari or fcn.")

flags.DEFINE_integer("episodes_between_updates", 1, "How many episodes to run before updating agent")
flags.DEFINE_bool("randomize_replay_buffer", True, "Randomize the replay buffer before updating an agent")
flags.DEFINE_bool("test_agent", True, "To run agent both in training and test mode")


FLAGS(sys.argv)
BASE_LOG_PATH = os.path.join(FLAGS.log_path, FLAGS.agent, str(FLAGS.step_mul))
# TRAIN_LOG = os.path.join(FLAGS.log_path, FLAGS.agent, str(FLAGS.step_mul), 'TRAIN')
# TEST_LOG = os.path.join(FLAGS.log_path, FLAGS.agent, str(FLAGS.step_mul), 'TEST')
# SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
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


def run_agent(agent, map_name, visualize, tb_training_writer, tb_testing_writer):
    start_time = time.time()
    with sc2_env.SC2Env(
            map_name=map_name,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        replay_buffer = []
        for episode_number in range(1, NUM_EPISODES+1):
            obs = env.reset()[0]  # Initial obs from env
            while True:
                prev_obs = obs
                action = agent.step(obs)
                obs = env.step([action])[0]
                s, a, r, s_ = agent.sc2obs_to_table_state(prev_obs), action.function, obs.reward, agent.sc2obs_to_table_state(obs)
                replay_buffer.append((s, a, r, s_))

                if obs.last():
                    should_update_agent = episode_number % FLAGS.episodes_between_updates == 0
                    if should_update_agent:
                        if FLAGS.randomize_replay_buffer:
                            random.shuffle(replay_buffer)
                        agent.update(replay_buffer)
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


def run(unused_argv):
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    sc_actions = [ScAction(actions.FUNCTIONS.no_op.id, False),
                  ScAction(actions.FUNCTIONS.Move_screen.id, True),
                  ScAction(actions.FUNCTIONS.Attack_screen.id, True)]

    if FLAGS.agent == "always_attack":
        agent = AlwaysAttackAgent()
    elif FLAGS.agent == "table":
        agent = TableAgent(learning_rate=FLAGS.learning_rate,
                           reward_decay=FLAGS.discount,
                           epsilon_greedy=0.9,
                           sc_actions=sc_actions)
    elif FLAGS.agent == "random":
        agent = RandomAgent(sc_actions=sc_actions)
    else:
        raise NotImplementedError()

    tb_training_writer = tf.summary.FileWriter(TRAIN_LOG)
    tb_testing_writer = tf.summary.FileWriter(TEST_LOG)

    run_agent(agent, FLAGS.map, FLAGS.render, tb_training_writer, tb_testing_writer)


if __name__ == "__main__":
    app.run(run)
