import absl.app as app
import pysc2.bin.agent as agent
import my_maps

my_maps.load_my_maps()

app.run(agent.main)
