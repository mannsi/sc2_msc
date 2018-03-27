from pysc2.maps import lib


class DefeatScv(lib.Map):
  directory = "mini_games"
  players = 1
  score_index = 0
  game_steps_per_episode = 0
  step_mul = 100  # This flag is never used.
  map_name = "DefeatScv"