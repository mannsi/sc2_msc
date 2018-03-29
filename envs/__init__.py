from gym.envs.registration import register
from envs.marine_vs_scv_envs import AttackOnly, AttackMoveNoop

register(
    id='MarineVsScvAttackOnly-v0',
    entry_point='envs:AttackOnly',
    kwargs={}
)

register(
    id='MarineVsScvAttackMoveNoop-v0',
    entry_point='envs:AttackMoveNoop',
    kwargs={}
)
