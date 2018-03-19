import logging

import absl.app as app
import pysc2.bin.agent

import my_maps
import my_log
import helper

try:
    file_log_level = int(helper.get_command_param_val('--file_log_level', remove_from_params=True))
    my_log.init_file_logging(file_log_level=file_log_level, file_name='logs/random_file_name.txt')
except ValueError:
    # No file logging param given
    pass

agent = helper.get_command_param_val('--agent', remove_from_params=False)
step_mul = helper.get_command_param_val('--step_mul', remove_from_params=False)

my_log.to_file(logging.INFO, f'STARTING. Agent:{agent}, Step_mul:{step_mul}')

my_maps.load_my_maps()
app.run(pysc2.bin.agent.main)
