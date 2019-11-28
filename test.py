import gym
import DCA_env
env = gym.make('multi-channel-DCA-v0')
done = False
env.reset()
while not done:
    action = env.action_space.sample()
    _, _, done, info = env.step(action)
    print(info)