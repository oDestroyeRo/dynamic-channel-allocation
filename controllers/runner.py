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
from stable_baselines import PPO2, DQN, A2C
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


class DCARunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
                env = Monitor(env, filename=monitor_path, allow_early_resets=True, info_keywords=('temp_blockprob','temp_total_blockprob','timestamp',))
                # Optionally, wrap the environment with the provided wrapper
                return env
            return _init

        n_envs = 12
        monitor_dir = "results"
        if self.args.model.upper() == "DQN":
            from stable_baselines.deepq.policies import MlpPolicy
            env = gym.make('multi-channel-DCA-v0')
            model = DQN(MlpPolicy, env=env, verbose=1, tensorboard_log='results/PPO', full_tensorboard_log=True, prioritized_replay=True, buffer_size=200000)
        elif self.args.model.upper() == "PPO":
            from stable_baselines.common.policies import MlpPolicy
            env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
            model = PPO2(MlpPolicy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10, ent_coef=0.0,
                learning_rate=3e-4, cliprange=0.2, verbose=2, tensorboard_log='results/PPO')
        elif self.args.model.upper() == "A2C":
            from stable_baselines.common.policies import MlpPolicy
            env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
            model = A2C(MlpPolicy, env=env, n_steps=256, verbose=2, learning_rate=3e-4, tensorboard_log='results/PPO')
        else:
            return

        model.learn(total_timesteps=200000000)
        model.save(self.log_dir + self.args.model.upper())

    def test(self):
        model = PPO2.load(self.log_dir + "A2C_15_200m")
        env = gym.make('multi-channel-DCA-v0')
        count = 0
        total_reward = 0
        for _ in tqdm(range(1100)):
            done = False
            state = env.reset()
            while not done:
                action, _ = model.predict(state)
                _, reward, done, info = env.step(action)
                count+=1
                total_reward += reward
                if info['is_nexttime']:
                    with open('results/a2c_15_200m_test.csv', 'a') as newFile:
                        newFileWriter = csv.writer(newFile)
                        print(info)
                        newFileWriter.writerow([total_reward, info['temp_blockprob'], info['temp_total_blockprob'], info['timestamp']])
                        total_reward = 0
        env.close()
