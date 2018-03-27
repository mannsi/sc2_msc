from gym.envs.registration import register
from envs.marine_vs_scv import MarineVsScvEnv

register(
    id='MarineVsScvEnv-v0',
    entry_point='envs:MarineVsScvEnv',
    kwargs={}
)


