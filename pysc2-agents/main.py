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

flags.DEFINE_integer("max_steps", 3, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("max_agent_steps", 100,
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
NUM_EPISODES = FLAGS.max_steps
if not os.path.exists(LOG):
    os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
    os.makedirs(SNAPSHOT)


def run_agent(agent, map_name, visualize):
    start_time = time.time()

    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:
        for episode_number in range(1, NUM_EPISODES+1):
            initial_obs = env.reset()[0]  # Initial obs from env
            episode = Episode(episode_number, MAX_AGENT_STEPS_PER_EPISODE, initial_obs)
            agent.reset()
            while True:
                action = agent.step(episode.current_obs.observation)
                obs = env.step([action])[0]
                episode.step(action, obs)
                if FLAGS.training:
                    if episode.done:
                        learning_rate = FLAGS.learning_rate * (1 - 0.9 * episode_number / NUM_EPISODES)
                        agent.update(list(episode.replay_buffer), FLAGS.discount, learning_rate, episode_number)
                        episode.print_cumulative_score()
                        if episode_number % FLAGS.snapshot_step == 1:
                            agent.save_model(SNAPSHOT, episode_number)
                        break  # Exit episode
                elif episode.done:
                    episode.print_cumulative_score()

    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


class Episode:
    def __init__(self, episode_number, max_agent_steps_per_episode, initial_obs):
        self.number = episode_number
        self.max_agent_steps_per_episode = max_agent_steps_per_episode
        self.initial_obs = initial_obs

        self.episode_step = 0
        self.is_done = False
        self.replay_buffer = []

    def step(self, action, current_obs):
        prev_obs = self.current_obs  # Get current obs before this step
        self.replay_buffer.append((prev_obs, action, current_obs))
        self.episode_step += 1

    def print_cumulative_score(self):
        print(f'Episode {self.number}, score: {self.current_obs.observation["score_cumulative"][0]}!')

    @property
    def current_obs(self):
        if self.episode_step > 0:
            current_obs = self.replay_buffer[-1][-1]  # current_obs of last replay buffer item
        else:
            current_obs = self.initial_obs
        return current_obs

    @property
    def done(self):
        return (self.episode_step >= self.max_agent_steps_per_episode) or self.current_obs.last()


def run(unused_argv):
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    # Setup agent
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    agent = agent_cls(FLAGS.training, FLAGS.screen_resolution)
    agent.build_model(False, DEVICE[0], FLAGS.net)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter(LOG)
    agent.initialize(sess, summary_writer)

    run_agent(agent, FLAGS.map, FLAGS.render)

if __name__ == "__main__":
    app.run(run)
