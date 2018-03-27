from tensorforce.agents import PPOAgent


def get_agent(env, network, saver):
    """
    Get ppo agent.
    :param env: OpenAIGym object
    :param network: list of layers as dict objects
    :return: tensorforce.agents.LearningAgent object
    """
    return PPOAgent(
        states=env.states,
        actions=env.actions,
        network=network,
        # batch_size=10,
        # Agent
        # preprocessing=None,
        # exploration=None,
        reward_preprocessing=None,
        saver=saver,
        # BatchAgent
        # keep_last_timestep=True,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-5
        ),
        optimization_steps=10,
        # Model
        scope='ppo',
        discount=0.99,
        # DistributionModel
        distributions=None,
        entropy_regularization=0.01,
        # PGModel
        baseline_mode="states",
        baseline={
            "type": "cnn",
            "conv_sizes": [32],
            "dense_sizes": [32]
        },
        baseline_optimizer={
            "type": "multi_step",
            "optimizer": {
                "type": "adam",
                "learning_rate": 1e-5
            },
            "num_steps": 10
        },
        gae_lambda=0.99,
        # normalize_rewards=False,
        # PGLRModel
        likelihood_ratio_clipping=0.2,
        # summary=None,
        # distributed=None
    )
