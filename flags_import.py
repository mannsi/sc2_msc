import sys
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
# flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
# flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
# flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("snapshot_step", 10, "Step for snapshot.")  # I use this to run the agent without training
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "/home/mannsi/Repos/sc2_msc/log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("minimap_resolution", 84, "Resolution for minimap feature layers.")

flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_integer("max_steps", 20, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84, "Resolution for screen feature layers.")
# flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_string("agent", "always_attack", "Which agent to run.")
flags.DEFINE_string("net", "atari", "atari or fcn.")


flags.DEFINE_integer("episodes_between_updates", 1, "How many episodes to run before updating agent")
flags.DEFINE_bool("randomize_replay_buffer", True, "Randomize the replay buffer before updating an agent")
flags.DEFINE_bool("save_replays", False, "If replays should be saved")
flags.DEFINE_bool("test_agent", True, "To run agent both in training and test mode")
flags.DEFINE_string("run_comment", "Normal", "A comment string to distinguish the run.")

FLAGS(sys.argv)


def get_flags():
    return FLAGS
