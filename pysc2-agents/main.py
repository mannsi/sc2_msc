from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

# noinspection PyUnresolvedReferences
import maps as my_maps

LOCK = threading.Lock()
FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
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

flags.DEFINE_integer("max_steps", 2, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
flags.DEFINE_integer("max_agent_steps", 60,
                     "Total agent steps.")  # Max agent steps per episode. Runs these steps or until it finishes
flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "atari", "atari or fcn.")

FLAGS(sys.argv)
if FLAGS.training:
    MAX_AGENT_STEPS_PER_EPISODE = FLAGS.max_agent_steps
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
else:
    MAX_AGENT_STEPS_PER_EPISODE = 1e5
    DEVICE = ['/cpu:0']

LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.net
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
MAX_NUM_EPISODES = FLAGS.max_steps
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)


def run_agent(agent, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        # Only for a single player!
        replay_buffer = []
        for recorder, episode_counter, is_done in run_episodes(agent, env, MAX_NUM_EPISODES, MAX_AGENT_STEPS_PER_EPISODE):
            if FLAGS.training:
                replay_buffer.append(recorder)
                if is_done:
                    # Learning rate schedule
                    learning_rate = FLAGS.learning_rate * (1 - 0.9 * episode_counter / MAX_NUM_EPISODES)
                    agent.update(replay_buffer, FLAGS.discount, learning_rate, episode_counter)
                    replay_buffer = []

                    obs = recorder[-1].observation
                    score = obs["score_cumulative"][0]
                    print('Your score is ' + str(score) + '!')

                    if episode_counter % FLAGS.snapshot_step == 1:
                        agent.save_model(SNAPSHOT, episode_counter)
            elif is_done:
                obs = recorder[-1].observation
                score = obs["score_cumulative"][0]
                print('Your score is ' + str(score) + '!')
        if FLAGS.save_replay:
            env.save_replay(agent.name)


def run_episodes(agent, env, num_episodes, max_steps_per_episode=0):
    """Agent and env interact via observations and actions"""
    start_time = time.time()
    try:
        for i in range(num_episodes):
            num_steps = 0
            timestep = env.reset()[0]
            agent.reset()
            episode_counter = i + 1
            while True:
                # This runs a single episode or until max frames are reached
                num_steps += 1
                prev_timestep = timestep
                action = agent.step(timestep.observation)
                timestep = env.step([action])[0]
                is_done = (num_steps >= max_steps_per_episode) or timestep.last()
                yield [prev_timestep, action, timestep], episode_counter, is_done
                if is_done:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % elapsed_time)


def _main(unused_argv):
    """Run agents"""
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    # Setup agents
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    agent = agent_cls(FLAGS.training, FLAGS.screen_resolution)
    agent.build_model(False, DEVICE[0], FLAGS.net)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(LOG)
    agent.setup(sess, summary_writer)

    agent.initialize()

    run_agent(agent, FLAGS.map, FLAGS.render)

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)
