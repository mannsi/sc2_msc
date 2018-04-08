from gym.envs.registration import register

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
