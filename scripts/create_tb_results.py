import os
import tensorflow as tf
import flags_import

FLAGS = flags_import.get_run_config()
BASE_LOG_PATH = os.path.join(FLAGS.log_path, "DM_FULLCONV", FLAGS.map, str(FLAGS.step_mul))

if FLAGS.run_comment is not "":
    BASE_LOG_PATH = os.path.join(BASE_LOG_PATH, FLAGS.run_comment)

# Have incremental log counter runs
log_counter = 0
while True:
    log_counter += 1
    run_log_path = os.path.join(BASE_LOG_PATH, str(log_counter))
    if not os.path.exists(run_log_path):
        break


tb_writer = tf.summary.FileWriter(run_log_path)

for i in range(1000):
    total_episode_rewards = 100
    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode rewards', simple_value=total_episode_rewards)])
    tb_writer.add_summary(reward_summary, i+1)

