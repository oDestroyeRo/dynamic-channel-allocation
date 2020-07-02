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


class DCARunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        self.args = args
        self.log_dir = "results/"


    def train(self):

        n_envs = 12
        monitor_dir = "results"
 
        env = make_vec_env('multi-channel-DCA-v0', n_envs=n_envs)

        if self.args.model.upper() == "DQN":
            from stable_baselines.deepq.policies import MlpPolicy
            env = gym.make('multi-channel-DCA-v0')
            # env = VecNormalize(env)
            model = DQN(MlpPolicy, env=env, verbose=1, tensorboard_log='results/RL', prioritized_replay=True, buffer_size=20000)
        elif self.args.model.upper() == "PPO":
            from stable_baselines.common.policies import MlpPolicy, CnnPolicy

            env = make_vec_env('multi-channel-DCA-v0', n_envs=n_envs)

            model = PPO2(CustomPolicy, env=env, n_steps=2096, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10, ent_coef=0.00,
                learning_rate=3e-4, cliprange=0.2, verbose=2, tensorboard_log='results/RL')
        elif self.args.model.upper() == "A2C":
            from stable_baselines.common.policies import MlpPolicy
            n_envs = 12
            env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
            # env = VecNormalize(env)
            model = A2C(CustomPolicy, env=env, n_steps=32, verbose=2, learning_rate=0.002, tensorboard_log='results/RL', vf_coef = 0.5, lr_schedule = 'linear', ent_coef = 0.0)
        elif self.args.model.upper() == "ACER":
            from stable_baselines.common.policies import MlpPolicy
            env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
            # env = VecNormalize(env)
            model = ACER(MlpPolicy, env=env, verbose=2, tensorboard_log='results/RL', ent_coef = 0.0, buffer_size = 100000)
        else:
            print("something wrong")
            return

        model.learn(total_timesteps=100000000)
        model.save(self.log_dir)

    def test(self):
        if self.args.model.upper() == "PPO":
            model = PPO2.load(self.log_dir + ".zip")
        elif self.args.model.upper() == "DQN":
            model = DQN.load(self.log_dir + ".zip")
        elif self.args.model.upper() == "A2C":
            model = A2C.load(self.log_dir + ".zip")
        env = gym.make('multi-channel-DCA-v0')
        total_reward = 0
        f = open("results/" + self.args.model.upper() + "/result.csv","w+")
        for _ in tqdm(range(8600)):
            done = False
            state = env.reset()
            count = 0
            total_utilization = 0
            while not done:
                if self.args.model.upper() == "RANDOM":
                    action = env.action_space.sample()
                elif self.args.model.upper() == "DCA":
                    state = np.reshape(state, (env.row, env.col, env.channels, env.status))
                    channels_avaliablel_list = np.arange(env.channels)
                    channels_avaliablel_list[:] = 0
                    
                    for i in range(env.channels):
                        channels_avaliablel_list[i] = len(np.where(state[:,:,i,0] == 0)[0])
                    action = np.where(channels_avaliablel_list == np.max(channels_avaliablel_list))[0][0]
                else:
                    action, _ = model.predict(state)
                _, reward, done, info = env.step(action)
                count+=1
                total_reward += reward
                total_utilization += info['utilization']
                if info['is_nexttime']:
                    f = open("results/" + self.args.model.upper() + "/result_sin_16_495.csv","a+")
                    newFileWriter = csv.writer(f)
                    print(info, total_utilization/count)
                    # newFileWriter.writerow([total_reward, info['temp_blockprob'], info['temp_total_blockprob'], info['drop_rate'], info['timestamp'], total_utilization/count])
                    newFileWriter.writerow([total_reward, info['temp_blockprob'], info['temp_total_blockprob'], info['drop_rate'], info['timestamp']])
                    total_reward = 0
                    count = 0
                    total_utilization = 0
        env.close()
