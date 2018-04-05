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
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf

import maps as my_maps

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
# flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("max_steps", 2, "Total steps for training.")  # Episodes
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

# flags.DEFINE_string("map","MoveToBeacon", "Name of a map to use.")
flags.DEFINE_string("map", "DefeatScv", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 84, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "atari", "atari or fcn.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 5,
                     "Total agent steps.")  # MANNSI: Is this num steps before agent updates weights?

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)
if FLAGS.training:
    PARALLEL = FLAGS.parallel
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
else:
    PARALLEL = 1
    MAX_AGENT_STEPS = 1e5
    DEVICE = ['/cpu:0']

LOG = FLAGS.log_path + FLAGS.map + '/' + FLAGS.net
SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)


def run_thread(agent, map_name, visualize):
    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        # env = available_actions_printer.AvailableActionsPrinter(env)

        # Only for a single player!
        replay_buffer = []
        for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):  # This runs run_loop only once
            if FLAGS.training:
                replay_buffer.append(recorder)
                if is_done:
                    with LOCK:
                        global COUNTER
                        COUNTER += 1
                        counter = COUNTER
                    # Learning rate schedule
                    learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
                    agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
                    replay_buffer = []

                    # MANNSI TODO: Does this work ???
                    obs = recorder[-1].observation
                    score = obs["score_cumulative"][0]
                    print('Your score is ' + str(score) + '!')

                    if counter % FLAGS.snapshot_step == 1:
                        agent.save_model(SNAPSHOT, counter)
                    if counter >= FLAGS.max_steps:
                        break  # This triggers finally cause in run loop function
            elif is_done:
                obs = recorder[-1].observation
                score = obs["score_cumulative"][0]
                print('Your score is ' + str(score) + '!')
        if FLAGS.save_replay:
            env.save_replay(agent.name)


def run_loop(agents, env, max_frames=0):
    """A run loop to have agents and an environment interact."""
    start_time = time.time()
    try:
        while True:
            num_frames = 0
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                num_frames += 1
                last_timesteps = timesteps
                actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
                timesteps = env.step(actions)
                # Only for a single player!
                is_done = (num_frames >= max_frames) or timesteps[0].last()
                yield [last_timesteps[0], actions[0], timesteps[0]], is_done
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

    agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
    agent.build_model(False, DEVICE[0], FLAGS.net)

    # agents = []
    # for i in range(PARALLEL):
    #     agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
    #     agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    #     agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(LOG)
    agent.setup(sess, summary_writer)
    # for i in range(PARALLEL):
    #     agents[i].setup(sess, summary_writer)

    agent.initialize()
    if not FLAGS.training or FLAGS.continuation:
        global COUNTER
        COUNTER = agent.load_model(SNAPSHOT)

    # Run threads
    # threads = []
    # for i in range(PARALLEL - 1):
    #     t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
    #     threads.append(t)
    #     t.daemon = True
    #     t.start()
    #     time.sleep(5)

    run_thread(agent, FLAGS.map, FLAGS.render)

    # for t in threads:
    #     t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)
