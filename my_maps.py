
from pysc2.maps import lib


class MyMap(lib.Map):
  directory = "mini_games"
  download = "https://github.com/deepmind/pysc2#get-the-maps"
  players = 1
  score_index = 0
  game_steps_per_episode = 0
  step_mul = 100  # This flag is never used.


def load_my_maps():
    """ This is how deep mind imports their maps"""
    mini_games = [
        "DefeatScv",
    ]

    for name in mini_games:
      globals()[name] = type(name, (MyMap,), dict(filename=name))
