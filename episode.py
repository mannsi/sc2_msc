class Episode:
    def __init__(self, episode_number, max_agent_steps_per_episode, initial_obs):
        self.number = episode_number
        self.max_agent_steps_per_episode = max_agent_steps_per_episode
        self.initial_obs = initial_obs

        self.steps = 0
        self.is_done = False
        self.replay_buffer = []

    def step(self, action, current_obs):
        prev_obs = self.current_obs  # Get current obs before this step
        self.replay_buffer.append((prev_obs, action, current_obs))
        self.steps += 1

    def print_cumulative_score(self):
        print(
            f'Episode {self.number}, score: {self.current_obs.observation["score_cumulative"][0]}, steps: {self.steps}')

    @property
    def current_obs(self):
        if self.steps > 0:
            current_obs = self.replay_buffer[-1][-1]  # current_obs of last replay buffer item
        else:
            current_obs = self.initial_obs
        return current_obs

    @property
    def done(self):
        return (self.steps >= self.max_agent_steps_per_episode) or self.current_obs.last()
