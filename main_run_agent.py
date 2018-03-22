import logging
import os

import absl.app as app
import pysc2.bin.agent

import my_maps
import my_log
import my_agents
import helper

# Init file logging
try:
    file_log_level = int(helper.get_command_param_val('--file_log_level', remove_from_params=True, default_val=logging.INFO))
    log_file_name = helper.get_command_param_val('--log_file_name', remove_from_params=True, default_val='random_file_name.txt')
    my_log.init_file_logging(file_log_level=file_log_level, file_name=os.path.join('logs', log_file_name))
except ValueError:
    # No file logging param given
    pass

# Log the run conditions
agent = helper.get_command_param_val('--agent', remove_from_params=False, default_val='my_agents.AttackAlwaysAgent')
step_mul = helper.get_command_param_val('--step_mul', remove_from_params=False, default_val=8)

# Dumb way to get my own cmd params to my agents
max_episodes = int(helper.get_command_param_val('--max_episodes', remove_from_params=True, default_val=0))
wait_after_attack = int(helper.get_command_param_val('--wait_after_attack', remove_from_params=True, default_val=0))
move_steps_after_attack = int(helper.get_command_param_val('--move_steps_after_attack', remove_from_params=True, default_val=1))
my_agents.GLOBAL_PARAM_MAX_EPISODES = max_episodes
my_agents.GLOBAL_WAIT_AFTER_ATTACK = wait_after_attack
my_agents.GLOBAL_MOVE_STEPS_AFTER_ATTACK = move_steps_after_attack

my_log.to_file(logging.WARNING, f'STARTING. Agent:{agent}, Step_mul:{step_mul}, Max_episodes:{max_episodes}, Wait_after_attack:{wait_after_attack}, Moves_after_attack:{move_steps_after_attack}')

# Init my map definitions
my_maps.load_my_maps()



# Run the agent
app.run(pysc2.bin.agent.main)
