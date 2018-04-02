from tensorforce.agents import RandomAgent


def get_agent(env, saver):
    """
    Get ppo agent.
    :param env: OpenAIGym object
    :param saver: Object to save model progress
    :return: tensorforce.agents.LearningAgent object
    """
    return RandomAgent(
        states=env.states,
        actions=env.actions,
        saver=saver)
