from maps.defeat_scv import DefeatScv


def load_my_maps():
    maps = [DefeatScv]

    for m in maps:
        globals()[m.map_name] = type(m.map_name, (m,), dict(filename=m.map_name))
