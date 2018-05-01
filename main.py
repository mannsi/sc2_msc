import os
import time

import tensorflow as tf
from absl import app

from pysc2.env import sc2_env
from pysc2.lib import stopwatch

# noinspection PyUnresolvedReferences
import maps as my_maps
from agents import SimpleVikingAgent
from models.basic_models import RandomModel, QLearningTableModel, PredefinedActionsModel, CmdInputModel, \
    HardCodedTableAgent

from sc2_action import Sc2Action
import constants
import flags_import


def run_agent(agent, run_config):
    """
    Run an agent using the given run config
    :param agent: The agent to run
    :type agent: agents.Sc2Agent
    :param run_config: The run configuration to use
    :type run_config: SimpleNamespace
    """
    start_time = time.time()
    replay_buffer = []
    step_counter = 0
    use_experience_replay = run_config.experience_replay_max_size is not None

    # CREATE SC2 ENV
    with sc2_env.SC2Env(
            map_name=run_config.map,
            step_mul=run_config.step_mul,
            screen_size_px=(run_config.screen_resolution, run_config.screen_resolution),
            minimap_size_px=(run_config.minimap_resolution, run_config.minimap_resolution),
            visualize=run_config.render,
            replay_dir=run_config.replay_dir,
            save_replay_episodes=run_config.eval_agent_steps) as env:
        for episode_number in range(1, run_config.max_steps + 1):
            obs = env.reset()[0]  # Initial obs from env
            while True:
                # INTERACT WITH ENVIRONMENT
                prev_obs = obs
                action = agent.act(obs)
                obs = env.step([action.get_function_call()])[0]
                s = agent.obs_to_state(prev_obs)
                a = action
                r = obs.reward
                s_ = agent.obs_to_state(obs) if not obs.last() else None

                # HANDLE EXPERIENCE REPLAY BUFFER
                if use_experience_replay:
                    step_counter += 1
                    if step_counter <= run_config.experience_replay_max_size:
                        # Start by filling the buffer
                        replay_buffer.append((s, a, r, s_))
                    else:
                        # Now systematically replace the oldest values
                        replay_buffer[step_counter % run_config.experience_replay_max_size] = (s, a, r, s_)
                else:
                    replay_buffer.append((s, a, r, s_))

                if obs.last():
                    # UPDATE AGENT
                    results_dict = agent.observe(replay_buffer)
                    agent.log_episode(obs, episode_number, results_dict)

                    if not use_experience_replay:
                        replay_buffer = []

                    # CONFIGURE IF NEXT EPISODE IS A TEST EPISODE
                    should_test_agent = agent.training_mode and episode_number % run_config.eval_agent_steps == 0
                    agent.training_mode = not should_test_agent

                    break  # Exit episode
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


def create_model(model_name, sc_actions, run_config):
    """

    :param model_name: Name of the model to create
    :type model_name: str
    :param sc_actions: The available actions of the model
    :type sc_actions: List of sc2_action.Sc2Action
    :param run_config: The run configuration
    :return: A model
    :rtype: sc2_model.Sc2Model
    """
    if model_name == "always_attack":
        model = RandomModel([Sc2Action(constants.ATTACK_ENEMY)])
    elif model_name == "random":
        model = RandomModel(sc_actions)
    elif model_name == "predefined_actions":
        action_list = [constants.ATTACK_ENEMY, constants.ATTACK_ENEMY, constants.FLIGHT] * 101
        model = PredefinedActionsModel(sc_actions, action_list)
    elif model_name == "hardcoded":
        model = HardCodedTableAgent(sc_actions)
    elif model_name == "cmd_input":
        key_to_action_mapping = {'f': constants.FLIGHT,
                                 'l': constants.LAND,
                                 'n': constants.NO_OP,
                                 'a': constants.ATTACK_ENEMY,
                                 'm': constants.MOVE_FROM_ENEMY}
        model = CmdInputModel(sc_actions, key_to_action_mapping)
    elif model_name == "table":
        model = QLearningTableModel(actions=sc_actions,
                                    lr=run_config.learning_rate,
                                    reward_decay=run_config.discount,
                                    epsilon_greedy=run_config.epsilon,
                                    total_episodes=run_config.max_steps,
                                    decay_lr=run_config.decay_lr,
                                    decay_epsilon=run_config.decay_epsilon)
    elif model_name == "1d_qlearning":
        num_inputs = run_config.screen_resolution * run_config.screen_resolution * 2  # 2 for our and enemy unit plains
        from models.dl_models import Dense1DModel  # Import here because importing TensorFlow is slow
        model = Dense1DModel(actions=sc_actions,
                             lr=run_config.learning_rate,
                             reward_decay=run_config.discount,
                             epsilon_greedy=run_config.epsilon,
                             total_episodes=run_config.max_steps,
                             num_inputs=num_inputs,
                             mini_batch_size=run_config.mini_batch_size,
                             log_dir=os.path.join(run_config.train_log_dir, 'model'))
    else:
        raise NotImplementedError()
    return model


def create_agent(sc_actions, run_config, model, log_dir, tb_train_writer, tb_test_writer):
    if run_config.agent == "simple_viking_agent":
        agent = SimpleVikingAgent(model, sc_actions, log_dir, tb_train_writer, tb_test_writer)
    else:
        raise NotImplementedError(f'Unknown agent {run_config.agent}')
    return agent


def save_run_configuration(run_config):
    sorted_conf_keys = sorted(run_config.__dict__)
    with open(os.path.join(run_config.run_log_path, 'run_config'), 'w') as f:
        for k in sorted_conf_keys:
            val = getattr(run_config, k)
            f.writelines(f'{k}: {val}\n')


def run(args):
    run_config = args[0]
    stopwatch.sw.enabled = run_config.profile or run_config.trace
    stopwatch.sw.trace = run_config.trace

    # TensorBoard loggers
    tb_test_writer = None
    tb_train_writer = tf.summary.FileWriter(run_config.train_log_dir)
    if run_config.test_agent:
        tb_test_writer = tf.summary.FileWriter(run_config.test_log_dir)

    save_run_configuration(run_config)

    sc_actions = [
        Sc2Action(constants.NO_OP),
        Sc2Action(constants.ATTACK_ENEMY),
        Sc2Action(constants.LAND),
        Sc2Action(constants.FLIGHT),
    ]

    model = create_model(run_config.model, sc_actions, run_config)
    agent = create_agent(run_config.agent, sc_actions, run_config, model, run_config.agent_files_dir, tb_train_writer, tb_test_writer)

    run_agent(agent, run_config)


if __name__ == "__main__":
    config = flags_import.get_run_config()
    app.run(run, argv=[config])
