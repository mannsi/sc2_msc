from tensorforce.agents import DQNAgent


def get_agent(env, network, saver):
    """
    Get ppo agent.
    :param env: OpenAIGym object
    :param network: list of layers as dict objects
    :param saver: Object to save model progress
    :return: tensorforce.agents.LearningAgent object
    """
    return DQNAgent(
        states=env.states,
        actions=env.actions,
        network=network,
        saver=saver
    )
