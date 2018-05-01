import sys
import os
from absl import flags
from pathlib import Path
from types import SimpleNamespace


FLAGS = flags.FLAGS

flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training.")
flags.DEFINE_float("discount", 0.9, "Discount rate for future rewards.")
flags.DEFINE_float("epsilon", 0.9, "Epsilon greedy parameter")
flags.DEFINE_integer("eval_agent_steps", 10, "Num of steps between agent eval")
flags.DEFINE_string("log_path", os.path.join(Path.home(), "tb_output"), "Path for log.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")

flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_integer("max_steps", 20, "Total steps for training.")  # Num episodes
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_string("agent", "simple_viking_agent", "Which agent to run.")
flags.DEFINE_string("model", "always_attack", "Which model to use.")

flags.DEFINE_integer("experience_replay_max_size", None, "Max steps to keep in replay buffer before overwriting")
flags.DEFINE_integer("mini_batch_size", 64, "Minibatch size")
flags.DEFINE_string("run_comment", "", "A comment string to distinguish the run.")
flags.DEFINE_bool("decay_lr", True, "If learning rate should be decayed or not")
flags.DEFINE_bool("decay_epsilon", True, "If epsilon greedy rate should be decayed or not")

FLAGS(sys.argv)


def get_run_config():
    extra_settings = {}

    agent_model_map = f'{FLAGS.agent}_{FLAGS.model}_{FLAGS.map}'
    BASE_LOG_PATH = os.path.join(FLAGS.log_path, agent_model_map)

    if FLAGS.run_comment is not "":
        BASE_LOG_PATH = os.path.join(BASE_LOG_PATH, FLAGS.run_comment)

    # Have incremental log counter runs
    log_counter = 0
    while True:
        log_counter += 1
        run_log_path = os.path.join(BASE_LOG_PATH, str(log_counter))
        if not os.path.exists(run_log_path):
            extra_settings['run_log_path'] = run_log_path

            train_log_dir = os.path.join(run_log_path, 'TRAIN')
            os.makedirs(train_log_dir)
            extra_settings['train_log_dir'] = train_log_dir

            if FLAGS.test_agent:
                test_log_dir = os.path.join(run_log_path, 'TEST')
                os.makedirs(test_log_dir)
                extra_settings['test_log_dir'] = test_log_dir
            break

    replay_dir = os.path.join(run_log_path, 'Replays')
    os.makedirs(replay_dir)
    extra_settings['replay_dir'] = replay_dir

    agent_files_dir = os.path.join(run_log_path, 'AgentFiles')
    os.makedirs(agent_files_dir)
    extra_settings['agent_files_dir'] = agent_files_dir

    joined_dict = {**extra_settings, **FLAGS.flag_values_dict()}
    return SimpleNamespace(**joined_dict)  # So I can access values by dot syntax
