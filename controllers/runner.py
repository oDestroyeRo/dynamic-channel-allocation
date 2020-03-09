import gym
import numpy as np
import csv
import os
from models.DQN import DQNAgent
import DCA_env
from datetime import datetime
import pytz
from pytz import timezone
from tqdm import tqdm
# from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLnLstmPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import register_policy
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

la = timezone("CET")

def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")

class SingleChannelRunner:
    def __init__(self, args):
        self.args = args
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="1"

    def train(self):
        env = gym.make('single-channel-DCA-v0')

        agent = DQNAgent(env.observation_space, env.action_space, self.args)

        # should be solved in this number of episodes
        episode_count = 100000000

        batch_size = 128

        timesteps = 0
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
                next_state, reward, done, info = env.step(action)
                next_state = np.expand_dims(next_state, 0)

                # store every experience unit in replay buffer
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                timesteps += 1
                count += 1
                total_block_prob += info['block_prob']
            if episode%10 == 0:
                #print(count, env.get_blockprop(), agent.epsilon, total_reward)
                with open('results/dqn_35_1_channel_4.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([timesteps, total_block_prob/count, total_reward/10, agent.epsilon])
                total_reward = 0
                total_block_prob = 0
                count = 0
            # call experience relay
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # close the env and write monitor result info to disk
        env.close() 
class MultiChannelRandom:
    def run(self):
        env = gym.make('multi-channel-DCA-v0')
        episode_count = 600
        for _ in tqdm(range(episode_count)):
            state = env.reset()
            done = False
            count = 0
            total_reward = 0
            while not done:
                # env.render()
                _, reward, done, info = env.step(env.action_space.sample())
                count+=1
                total_reward += reward
            with open('results/random.csv', 'a') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([total_reward, count, info['block_prob'], info['timestamp']])
        env.close()

class MultiChannelPPORunner:
    def __init__(self, args):
        import os
        # os.environ["CUDA_VISIBLE_DEVICES"]="1"
        self.args = args
        self.log_dir = "results/"


    def train(self):
        def make_env(rank,env_id,monitor_dir):
            def _init():
                env = gym.make(env_id)
                # Wrap the env in a Monitor wrapper
                # to have additional training information
                monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
                # Create the monitor folder if needed
                if monitor_path is not None:
                    os.makedirs(monitor_dir, exist_ok=True)
                env = Monitor(env, filename=monitor_path, allow_early_resets=True, info_keywords=('block_prob','timestamp',))
                # Optionally, wrap the environment with the provided wrapper
                return env
            return _init
        # policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 256])
        # env = gym.make('multi-channel-DCA-v0')
        # env = gym.make('multi-channel-DCA-v0')
        # space = env.observation_space
        # print(env.observation_space.shape)
        # env = make_vec_env('multi-channel-DCA-v0', n_envs=4, monitor_dir="results")
        # n_cpu = 12
        n_envs = 6
        monitor_dir = "results"
        # env = SubprocVecEnv([lambda: gym.make('multi-channel-DCA-v0') for i in range(n_cpu)])
        # env = Monitor(env, self.log_dir, allow_early_resets=True, info_keywords=('block_prob','timestamp',))
        env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
        # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 128])
        # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 256, 128])
        # model = PPO2(MlpPolicy, env, verbose=1, gamma=0.99, n_steps=512, nminibatches=128)
        # model = PPO2(MlpPolicy, env, verbose=1, gamma=0.99, n_steps=128, nminibatches=4, cliprange=0.2)
        # model = PPO2("MlpPolicy", env, verbose=1)

        model = PPO2(CustomPolicy, env=env, n_steps=1024, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, verbose=2, tensorboard_log='results/PPO')
        model.learn(total_timesteps=100000000)
        model.save(self.log_dir + "ppo2_multi")

    def test(self):
        model = PPO2.load(self.log_dir + "ppo2_multi")
        env = gym.make('multi-channel-DCA-v0')
        state = env.reset()
        # episode_count = 100
        # for _ in tqdm(range(episode_count)):
        state = env.reset()
        done = False
        count = 0
        total_reward = 0
        while not done:
            # env.render()
            # action, _ = model.predict(state)
            _, reward, done, info = env.step(env.action_space.sample())
            count+=1
            total_reward += reward
            if info['is_nexttime']:
                with open('results/ppo_real_traffic_10_test_random.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([total_reward, info['temp_blockprob'], info['timestamp']])
                    total_reward = 0
        env.close()

class MultiChannelRunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.args = args

    def train(self):
        env = gym.make('multi-channel-DCA-v0')
        # env = Monitor(env, './results/videos', force=True)

        agent = DQNAgent(env.observation_space, env.action_space, self.args)

        # should be solved in this number of episodes
        episode_count = 1000000

        batch_size = 128

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
                    newFileWriter.writerow([count, total_block_prob/5, total_reward/5, agent.epsilon, datetime.fromtimestamp(max_timestamp, la).strftime('%Y-%m-%d %H:%M:%S')])
                total_reward = 0
                total_block_prob = 0
                max_timestamp = 0
                # env.blocktimes = 0
                # env.timestep = 1
            # call experience relay
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # close the env and write monitor result info to disk
        env.close() 
