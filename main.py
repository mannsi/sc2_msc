from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

import tensorflow as tf
from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import TableAgent, AlwaysAttackAgent

LOCK = threading.Lock()
FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
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

FLAGS(sys.argv)
LOG = os.path.join(FLAGS.log_path, FLAGS.agent)
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
NUM_EPISODES = FLAGS.max_steps
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)


def run_agent(agent, map_name, visualize, tb_writer):
    start_time = time.time()
    with sc2_env.SC2Env(
            map_name=map_name,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        for episode_number in range(1, NUM_EPISODES+1):
            replay_buffer = []
            obs = env.reset()[0]  # Initial obs from env
            while True:
                prev_obs = obs
                action = agent.step(obs)
                obs = env.step([action])[0]
                replay_buffer.append((prev_obs, action, obs.reward, obs))

                if obs.last():
                    agent.update(replay_buffer)
                    log_episode(tb_writer, obs, episode_number)
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

    if FLAGS.agent == "always_attack":
        agent = AlwaysAttackAgent()
    elif FLAGS.agent == "table":
        agent = TableAgent(x_size=FLAGS.screen_resolution,
                           y_size=FLAGS.screen_resolution,
                           step_size=FLAGS.learning_rate,
                           discount=FLAGS.discount)
    else:
        raise NotImplementedError()

    tb_writer = tf.summary.FileWriter(LOG)

    run_agent(agent, FLAGS.map, FLAGS.render, tb_writer)

if __name__ == "__main__":
    app.run(run)
