import constants


def get_player_relative_view(obs):
    """ View from player camera perspective. Returns an NxN np array """
    return obs.observation["screen"][constants.PLAYER_RELATIVE]


def get_own_unit_location(obs):
    """ Mean values of friendly unit coordinates, returned as a (x,y) tuple """
    own_unit_loc_y, own_unit_loc_x = get_own_unit_locations(obs)
    return own_unit_loc_x.mean(), own_unit_loc_y.mean()


def get_own_unit_locations(obs):
    """ My own unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
    return (get_player_relative_view(obs) == constants.UNITS_MINE).nonzero()


def get_enemy_unit_location(obs):
    """ Mean values of enemy unit coordinates, returned as a (x,y) tuple """
    enemy_unit_loc_y, enemy_unit_loc_x = get_enemy_unit_locations(obs)
    return enemy_unit_loc_x.mean(), enemy_unit_loc_y.mean()


def get_enemy_unit_locations(obs):
    """ Enemy unit locations as a tuple of (np_array_of_Y_locations, np_array_of_X_locations)"""
    return (get_player_relative_view(obs) == constants.UNITS_ENEMY).nonzero()
