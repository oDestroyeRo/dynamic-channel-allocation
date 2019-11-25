import gym
import numpy as np
import csv
from models.DQN import DQNAgent
import DCA_env
from datetime import datetime
from tqdm import tqdm
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, HER, DQN, SAC, DDPG, TD3, ACKTR, ACER, A2C, TRPO
from stable_baselines.bench import Monitor
import tensorflow as tf



class SingleChannelRunner:
    def __init__(self, args):
        self.args = args

    def train(self):
        env = gym.make('single-channel-DCA-v0')
        state_size = env.observation_space
        action_size = env.action_space

        agent = DQNAgent(env.observation_space, env.action_space, self.args)

        # should be solved in this number of episodes
        episode_count = 100000000

        batch_size = 64

        count = 0
        total_reward = 0
        total_block_prob = 0
        # Q-Learning sampling and fitting
        for episode in tqdm(range(episode_count)):
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

class MultiChannelPPORunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.args = args
        self.log_dir = "tmp/"
    def train(self):
        # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256])
        env = gym.make('multi-channel-DCA-v0')
        env = Monitor(env, self.log_dir)
        env = DummyVecEnv([lambda: env])
        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=1000000000)
        model.save(self.log_dir + "ppo2_multi")

    def test(self):
        model = PPO2.load(self.log_dir + "ppo2_multi")
        episode_count = 100000
        count = 0
        total_reward = 0
        total_block_prob = 0
        max_timestamp = 0
        timestamp = 0
        env = gym.make('multi-channel-DCA-v0')
        state = env.reset()
        for episode in tqdm(range(episode_count)):
            state = env.reset()
            done = False
            while not done:
                action, _ = model.predict(state)
                next_state, reward, done, _ = env.step(action)
                timestamp = env.get_timestamp()
                if (timestamp > max_timestamp):
                    max_timestamp = timestamp
                state = next_state
                total_reward += reward
                count += 1
            total_block_prob += env.get_blockprob()
            if episode%100 == 0:
                with open('results/ppo_35_multi_channel_real_traffic.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([count, total_block_prob/100, total_reward/100, datetime.utcfromtimestamp(max_timestamp).strftime('%Y-%m-%d %H:%M:%S')])
                total_reward = 0
                total_block_prob = 0
                env.blocktimes = 0
                env.timestep = 1
        env.close()

class MultiChannelRunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.args = args

    def train(self):
        env = gym.make('multi-channel-DCA-v0')
        state_size = env.observation_space
        action_size = env.action_space

        agent = DQNAgent(env.observation_space, env.action_space, self.args)

        # should be solved in this number of episodes
        episode_count = 1000000

        batch_size = 64

        count = 0
        total_reward = 0
        total_block_prob = 0
        max_timestamp = 0
        timestamp = 0
        for episode in range(episode_count):
            state = env.reset()

            state = np.expand_dims(state, 0)
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                timestamp = env.get_timestamp()
                if (timestamp > max_timestamp):
                    max_timestamp = timestamp
                next_state = np.expand_dims(next_state, 0)

                # store every experience unit in replay buffer
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                count += 1
            total_block_prob += float(info['blockprob'])
            if episode%5 == 0:
                #print(count, env.get_blockprop(), agent.epsilon, total_reward)
                with open('results/dqn_35_multi_channel_real_traffic.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([count, total_block_prob/5, total_reward/5, agent.epsilon, datetime.utcfromtimestamp(max_timestamp).strftime('%Y-%m-%d %H:%M:%S')])
                total_reward = 0
                total_block_prob = 0
                # env.blocktimes = 0
                # env.timestep = 1
            # call experience relay
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # close the env and write monitor result info to disk
        env.close() 
