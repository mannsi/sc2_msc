from pysc2.lib import features


# Internal action ids
NO_OP = "_NO_OP"
ATTACK_ENEMY = "_ATTACK_ENEMY"
MOVE_TO_ENEMY = "_MOVE_TO_ENEMY"
MOVE_FROM_ENEMY = "_MOVE_FROM_ENEMY"
FLIGHT = "_FLIGHT"
LAND = "_LAND"


# Screen features
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

# define constants about AI's world
UNITS_MINE = 1
UNITS_ENEMY = 4

NOT_QUEUED = [0]
