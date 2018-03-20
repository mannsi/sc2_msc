import absl.app as app
import pysc2.bin.play

import my_maps

# Init my map definitions
my_maps.load_my_maps()

# Run the agent
app.run(pysc2.bin.play.main)
