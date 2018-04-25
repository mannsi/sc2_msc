from pysc2.maps import lib


class DefeatScv(lib.Map):
    directory = "mini_games"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 100  # This flag is never used.
    map_name = "DefeatScv"


class DefeatLing(lib.Map):
    directory = "mini_games"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 100  # This flag is never used.
    map_name = "DefeatLing"


class VikingVsBaneling(lib.Map):
    directory = "mini_games"
    map_name = "VikingVsBaneling"


def load_my_maps():
    maps = [DefeatScv, DefeatLing, VikingVsBaneling]

    for m in maps:
        globals()[m.map_name] = type(m.map_name, (m,), dict(filename=m.map_name))
