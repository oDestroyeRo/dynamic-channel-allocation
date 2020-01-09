from gym.envs.registration import register

register(
    id='single-channel-DCA-v0',
    entry_point='DCA_env.envs:SingleChannelDCAEnv',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

register(
    id='multi-channel-DCA-v0',
    entry_point='DCA_env.envs:MultiChannelDCAEnv',
    # max_episode_steps=1000,
    reward_threshold=1.0,
)
