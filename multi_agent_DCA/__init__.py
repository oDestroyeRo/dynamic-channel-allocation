from gym.envs.registration import register

register(
    id='multi-agent-DCA-v0',
    entry_point='multi_agent_DCA.envs:MultiAgentDCAEnv',
)