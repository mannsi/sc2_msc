from tensorforce.agents import Agent
from tensorforce.models.random_model import RandomModel

import envs.marine_vs_scv

_UNITS_MINE = 1
_UNITS_ENEMY = 4
_ATTACK_SCREEN_ACTION_INDEX = envs.marine_vs_scv._ATTACK_SCREEN_ACTION_INDEX


def get_agent(env, saver):
    """
    Get ppo agent.
    :param env: OpenAIGym object
    :param saver: Object to save model progress
    :return: tensorforce.agents.LearningAgent object
    """
    return AlwaysAttackScv(
        states=env.states,
        actions=env.actions,
        saver=saver)


class AlwaysAttackScv(Agent):
    def __init__(
        self,
        states,
        actions,
        batched_observe=True,
        batching_capacity=1000,
        scope='random',
        device=None,
        saver=None,
        summarizer=None,
    ):
        """
        Initializes the random agent.

        Args:
            scope (str): TensorFlow scope (default: name of agent).
            device: TensorFlow device (default: none)
            saver (spec): Saver specification, with the following attributes (default: none):
                - directory: model directory.
                - file: model filename (optional).
                - seconds or steps: save frequency (default: 600 seconds).
                - load: specifies whether model is loaded, if existent (default: true).
                - basename: optional file basename (default: 'model.ckpt').
            summarizer (spec): Summarizer specification, with the following attributes (default:
                none):
                - directory: summaries directory.
                - seconds or steps: summarize frequency (default: 120 seconds).
                - labels: list of summary labels to record (default: []).
                - meta_param_recorder_class: ???.
            execution (spec): Execution specification (see sanity_check_execution_spec for details).
        """

        self.scope = scope
        self.device = device
        self.saver = saver
        self.summarizer = summarizer

        super().__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity
        )

    def act(self, states, deterministic=False, independent=False, fetch_tensors=None):
        """
        Return action(s) for given state(s). States preprocessing and exploration are applied if
        configured accordingly.

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            deterministic (bool): If true, no exploration and sampling is applied.
            independent (bool): If true, action is not followed by observe (and hence not included
                in updates).
            fetch_tensors (list): Optional String of named tensors to fetch
        Returns:
            Scalar value of the action or dict of multiple actions the agent wants to execute.
            (fetched_tensors) Optional dict() with named tensors fetched
        """
        enemy_location = self._get_enemy_unit_location(states)

        self.current_actions = {'action0':_ATTACK_SCREEN_ACTION_INDEX, 'action1': enemy_location[0], 'action2':enemy_location[1]}
        return self.current_actions

        # BELOW IS THE DEFAULT CODE FOR AN AGENT
        # self.current_internals = self.next_internals
        #
        # if self.unique_state:
        #     self.current_states = dict(state=np.asarray(states))
        # else:
        #     self.current_states = {name: np.asarray(state) for name, state in states.items()}
        #
        # if fetch_tensors is not None:
        #     # Retrieve action
        #     self.current_actions, self.next_internals, self.timestep, self.fetched_tensors = self.model.act(
        #         states=self.current_states,
        #         internals=self.current_internals,
        #         deterministic=deterministic,
        #         independent=independent,
        #         fetch_tensors=fetch_tensors
        #     )
        #
        #     if self.unique_action:
        #         return self.current_actions['action'], self.fetched_tensors
        #     else:
        #         return self.current_actions, self.fetched_tensors
        #
        # else:
        #     # Retrieve action
        #     self.current_actions, self.next_internals, self.timestep = self.model.act(
        #         states=self.current_states,
        #         internals=self.current_internals,
        #         deterministic=deterministic,
        #         independent=independent
        #     )
        #
        #     if self.unique_action:
        #         return self.current_actions['action']
        #     else:
        #         return self.current_actions

    def _get_enemy_unit_location(self, states):
        """ Mean values of enemy unit coordinates, returned as a (x,y) tuple """
        enemy_unit_loc_y, enemy_unit_loc_x = (states == _UNITS_ENEMY).nonzero()[1:]
        return enemy_unit_loc_x.mean(), enemy_unit_loc_y.mean()

    def observe(self, terminal, reward):
        """
        Observe experience from the environment to learn from. Optionally pre-processes rewards
        Child classes should call super to get the processed reward
        EX: terminal, reward = super()...

        Args:
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.
        """
        # BELOW IS THE DEFAULT CODE FOR AN AGENT UNCHANGED
        self.current_terminal = terminal
        self.current_reward = reward

        if self.batched_observe:
            # Batched observe for better performance with Python.
            self.observe_terminal.append(self.current_terminal)
            self.observe_reward.append(self.current_reward)

            if self.current_terminal or len(self.observe_terminal) >= self.batching_capacity:
                self.episode = self.model.observe(
                    terminal=self.observe_terminal,
                    reward=self.observe_reward
                )
                self.observe_terminal = list()
                self.observe_reward = list()

        else:
            self.episode = self.model.observe(
                terminal=self.current_terminal,
                reward=self.current_reward
            )

    def initialize_model(self):
        return RandomModel(
            states=self.states,
            actions=self.actions,
            scope=self.scope,
            device=self.device,
            saver=self.saver,
            summarizer=self.summarizer,
            execution=None,
            batching_capacity=self.batching_capacity
        )
