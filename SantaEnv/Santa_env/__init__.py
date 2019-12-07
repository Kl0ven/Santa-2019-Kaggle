from gym.envs.registration import register
register(
    id='Santa-v0',
    entry_point='Santa_env.envs:Santa_env',
)
