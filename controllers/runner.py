import gym
import numpy as np
import csv
import os
import os.path
from models.DQN import DQNAgent
import DCA_env
from datetime import datetime
import pytz
from pytz import timezone
from tqdm import tqdm
# from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLnLstmPolicy
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, DQN, A2C, ACER
from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import register_policy
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import FeedForwardPolicy, register_policy


la = timezone("CET")

def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = conv_to_fc(layer_3)
    layer_5 = linear(layer_4, 'fc1', n_hidden=1024, init_scale=np.sqrt(2))
    layer_6 = linear(layer_5, 'fc2', n_hidden=512, init_scale=np.sqrt(2))
    return activ(layer_6)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn, feature_extraction="cnn")


class DCARunner:
    def __init__(self, args):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        self.args = args
        self.log_dir = "results/" + self.args.model.upper() + "/" + self.args.model.upper()


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

        n_envs = 16
        monitor_dir = "results"
        if self.args.model.upper() == "DQN":
            from stable_baselines.deepq.policies import MlpPolicy
            env = gym.make('multi-channel-DCA-v0')
            # env = VecNormalize(env)
            model = DQN(MlpPolicy, env=env, verbose=1, tensorboard_log='results/RL', prioritized_replay=True, buffer_size=20000)
        elif self.args.model.upper() == "PPO":
            from stable_baselines.common.policies import MlpPolicy, CnnPolicy
            n_envs = 12
            # env = DummyVecEnv([make_env(i, 'multi-channel-DCA-v0', monitor_dir) for i in range(n_envs)])
            env = make_vec_env('multi-channel-DCA-v0', n_envs=n_envs)
            # env = VecNormalize(env)
            env = VecFrameStack(env, n_stack=3)
            model = PPO2(CustomPolicy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10, ent_coef=0.0,
                learning_rate=2.5e-4, cliprange=0.2, verbose=2, tensorboard_log='results/RL')
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

        model.learn(total_timesteps=200000000)
        model.save(self.log_dir)

    def test(self):
        if self.args.model.upper() == "PPO":
            model = PPO2.load(self.log_dir + ".zip")
        elif self.args.model.upper() == "DQN":
            model = DQN.load(self.log_dir + ".zip")
        elif self.args.model.upper() == "A2C":
            model = A2C.load(self.log_dir + ".zip")
        env = gym.make('multi-channel-DCA-v0')
        count = 0
        total_reward = 0
        f = open("results/" + self.args.model.upper() + "/result.csv","w+")
        for _ in tqdm(range(8600)):
            done = False
            state = env.reset()
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
                if info['is_nexttime']:
                    f = open("results/" + self.args.model.upper() + "/result_3.csv","a+")
                    newFileWriter = csv.writer(f)
                    print(info)
                    newFileWriter.writerow([total_reward, info['temp_blockprob'], info['temp_total_blockprob'], info['drop_rate'], info['timestamp']])
                    total_reward = 0
        env.close()
