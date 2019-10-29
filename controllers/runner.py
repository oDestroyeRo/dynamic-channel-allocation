import gym
import numpy as np
import csv
from models.DQN import DQNAgent
import DCA_env


class SingleChannelRunner:
    def train(self):
        env = gym.make('single-channel-DCA-v0')
        state_size = env.observation_space
        action_size = env.action_space

        agent = DQNAgent(env.observation_space, env.action_space)

        # should be solved in this number of episodes
        episode_count = 1000000

        batch_size = 64

        count = 0
        total_reward = 0
        total_block_prob = 0
        # Q-Learning sampling and fitting
        for episode in range(episode_count):
            state = env.reset()

            state = np.expand_dims(state, 0)
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, 0)

                # store every experience unit in replay buffer
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                count += 1
            total_block_prob += env.get_blockprob()
            if episode%100 == 0:
                #print(count, env.get_blockprop(), agent.epsilon, total_reward)
                with open('results/dqn_35_1_channel.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([count, total_block_prob/100, total_reward/100, agent.epsilon])
                total_reward = 0
                total_block_prob = 0
                env.blocktimes = 0
                env.timestep = 0
            # call experience relay
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # close the env and write monitor result info to disk
        env.close() 