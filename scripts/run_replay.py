import absl.app as app
import pysc2.bin.play

# noinspection PyUnresolvedReferences
import maps as my_maps

# Hack required because master version of pysc2 does not support 4.0.2 version
import pysc2.run_configs.platforms as platforms
game_version = "4.0.2"
if game_version not in platforms.VERSIONS:
    platforms.VERSIONS[game_version] = (59877, "B43D9EE00A363DAFAD46914E3E4AF362")


# Run the agent
app.run(pysc2.bin.play.main)
