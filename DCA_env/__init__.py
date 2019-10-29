from gym.envs.registration import register

register(
    id='single-channel-DCA-v0',
    entry_point='DCA_env.envs:SingleChannelDCAEnv',
)

register(
    id='multi-channel-DCA-v0',
    entry_point='DCA_env.envs:MultiChannelDCAEnv',
)
