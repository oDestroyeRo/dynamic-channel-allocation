import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
import math
from datetime import datetime
from pytz import timezone
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import random
# from multi_discrete import MultiDiscrete

la = timezone("CET")


class MultiChannelDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        bs_datas = pd.read_csv('Milano_bs.csv')
        self.bs_position = bs_datas[['lat','lon']]
        self.bs_position = self.bs_position.sort_values(by=['lat', 'lon'])
        self.bs_range = 0.0045045045 # 500m
        self.list_bs_in_range = dict()
        self.traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
        self.row = 12
        self.col = 12
        # self.number_bs = 143
        # self.traffic_channel = 500
        # self.channels = np.max(self.traffic_data[:,:,:,1]) / self.traffic_channel
        # self.channels = math.ceil(self.channels)
        # self.channels = int(self.channels)
        self.channels = 100
        self.status = 2 #channel available //location
        self.current_base_station = [0,0]
        self.reward = 0
        self.timestep = 1
        self.blocktimes = 0
        self.state = None
        self.traffic_timestep = 0
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]
        self.queue = 0
        self.action_space = spaces.Discrete(self.channels)
        self.observation_space = spaces.Box(low=0, high=self.channels, shape=(self.row ,self.col ,self.channels *self.status), dtype=np.uint64)

        self.viewer = None
        self.seed()

        self.array_render = np.zeros([self.row, self.col], dtype=object)

        for i in range(143):
            tmp = []
            for j in range(143):
                distance = math.sqrt(pow(self.bs_position.iloc[i,0] - self.bs_position.iloc[j,0],2) + pow(self.bs_position.iloc[i,1] - self.bs_position.iloc[j,1],2))
                if distance < self.bs_range and distance > 0:
                    tmp.append(j)
            self.list_bs_in_range[i] = tmp



    
    # def check_dca(self, action, state):
    #     c_bs_r = self.current_base_station[0]
    #     c_bs_c = self.current_base_station[1]
    #     if action+1 in set(state[c_bs_r, c_bs_c, :, 0]):
    #         return False
    #     if c_bs_r != 0 and action+1 in set(state[c_bs_r-1, c_bs_c, :, 0]):
    #         return False
    #     if c_bs_r != self.row-1 and action+1 in set(state[c_bs_r+1, c_bs_c, :, 0]):
    #         return False
    #     if c_bs_c != 0 and action+1 in set(state[c_bs_r, c_bs_c-1, :, 0]):
    #         return False
    #     if c_bs_c != self.col-1 and action+1 in set(state[c_bs_r, c_bs_c+1, :, 0]):
    #         return False
    #     if c_bs_r != self.row-1 and c_bs_c != 0 and action+1 in set(state[c_bs_r+1, c_bs_c-1, :, 0]):
    #         return False
    #     if c_bs_r != 0 and c_bs_c != self.col-1 and action+1 in set(state[c_bs_r-1, c_bs_c+1, :, 0]):
    #         return False
    #     return True

    def check_dca_real_bs(self, action, state):
        cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
        if cur_index >= 143:
            cur_index = 142
        neighbor = self.list_bs_in_range[cur_index]
        for i in range(len(neighbor)):
            row = i // 12
            col = i % 12
            if action in state[row, col, :, 0]:
                return False
        return True

    def is_channel_avalable(self, state):
        used_channel = set([])
        cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
        if cur_index >= 143:
            cur_index = 142
        neighbor = self.list_bs_in_range[cur_index]
        for i in range(len(neighbor)):
            row = i // 12
            col = i % 12
            used_channel.update(set(state[row, col, :, 0]))
        if len(used_channel) >= self.channels:
            return False
        return True

    # def is_next_bs(self,state):
    #     self.status_array[self.current_base_station[0], self.current_base_station[1], 0] -= 10
    #     if int(self.status_array[self.current_base_station[0], self.current_base_station[1]]) <= 0:
    #         state = self.next_bs(state)
    #     else:
    #         state = self.next_channel(state)

    #     return state


    def next_request(self, state):
        # self.queue = 0
        self.status_array[self.current_base_station[0], self.current_base_station[1], 0] -= 10
        state[self.current_base_station[0], self.current_base_station[1], :, 1] = 0
        queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])
        if int(self.status_array[self.current_base_station[0], self.current_base_station[1], 0]) <= 0 or queue >= self.channels:
            cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
            self.blocktimes += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 10
            self.timestep += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 10
            self.bs_available.remove(cur_index)
        if len(self.bs_available) > 0:
            random_index = np.random.randint(len(self.bs_available))
            bs_random_index = self.bs_available[random_index]
            self.current_base_station[0] = bs_random_index // self.row
            self.current_base_station[1] = bs_random_index % self.col
            while not self.is_channel_avalable(state):
                # print(self.current_base_station)
                self.blocktimes += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 10
                self.timestep += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 10
                self.status_array[self.current_base_station[0], self.current_base_station[1], 0] = 0
                # cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
                self.bs_available.remove(bs_random_index)
                if (len(self.bs_available) <= 0):
                    break
                random_index = np.random.randint(len(self.bs_available))
                bs_random_index = self.bs_available[random_index]
                self.current_base_station[0] = bs_random_index // self.row
                self.current_base_station[1] = bs_random_index % self.col
        # print(bs_random_index, self.bs_available, random_index)

        if (len(self.bs_available) <= 0):
            # self.blocktimes += np.sum(self.status_array[:,:,1])
            self.traffic_timestep += 1
            if self.traffic_timestep - self.temp_timestep >= 5:
                self.done = True
            self.set_timestamp()
            state = np.zeros([self.row, self.col, self.channels, self.status], dtype=np.uint64)
            
            self.status_array = np.zeros((self.row,self.col,2))
            for i in range(self.row):
                for j in range(self.col):
                    self.status_array[i,j,0] = self.traffic_data[self.traffic_timestep, i, j, 1]
            self.bs_available = []
            for i in range(144):
                self.bs_available.append(i)
        queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])
        state[self.current_base_station[0], self.current_base_station[1], queue, 1] = self.channels

        return state
        




        # self.current_base_station[1] += 1
        # if self.current_base_station[1] >= self.col:
        #     self.current_base_station[0] += 1
        #     if (self.current_base_station[0] >= self.row and self.current_base_station[1] >= self.col):
        #         # self.done = True
        #         self.traffic_timestep += 1
        #         if self.traffic_timestep >= self.traffic_data.shape[0]:
        #             self.reward = 0
        #             self.done = True
        #         else:
        #             self.set_timestamp()
        #             state = np.zeros([self.row, self.col, self.channels, self.status], dtype=np.uint32)
        #             for i in range(self.row):
        #                 for j in range(self.col):
        #                     # state[i, j, :, 0] = self.channels+1
        #                     state[i, j, :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, i, j, 1] / self.traffic_channel)
        #         if self.traffic_timestep - self.temp_timestep >= 6:
        #             self.done = True
        #     self.current_base_station[1] = 0
        #     if self.current_base_station[0] >= self.row:
        #         self.current_base_station[0] = 0
        # # state[self.current_base_station[0], self.current_base_station[1], :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] / self.traffic_channel)
        # state[self.current_base_station[0], self.current_base_station[1], self.queue, 1] = self.channels + 1
        # state[self.current_base_station[0], self.current_base_station[1], :, 0] = 0
        # return state

    # def next_channel(self, state):
    #     state[self.current_base_station[0], self.current_base_station[1], self.queue-1, 1] = 0
    #     state[self.current_base_station[0], self.current_base_station[1], self.queue, 1] = self.channels + 1
    #     return state

    def step(self, action):
        action = action + 1
        # print(self.current_base_station)
        self.done = False
        state = self.state
        # print(self.current_base_station, int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1]))
        # print(state[self.current_base_station[0],self.current_base_station[1],:,1])
        # print(state[self.current_base_station[0],self.current_base_station[1],:,2])
        # if not self.check_channel_avalable(state):
        #     # self.reward = -int(state[self.current_base_station[0], self.current_base_station[1], action, 2])
        #     self.reward = 0
        #     self.blocktimes += 1
        #     state[self.current_base_station[0], self.current_base_station[1], :, 2] -= 1
        #     state = self.next_bs(state)
        #     self.timestep +=1
        if self.check_dca_real_bs(action, state):
            # self.reward = -int(state[self.current_base_station[0], self.current_base_station[1], action, 2])
            # self.reward = 1/int(state[self.current_base_station[0], self.current_base_station[1], action, 2])
            self.reward = 1
            # state[self.current_base_station[0], self.current_base_station[1], :, 2] -= 1
            queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])

            state[self.current_base_station[0], self.current_base_station[1], queue, 0] = action

            # self.status_array[self.current_base_station[0], self.current_base_station[1], 0] -= 10

            # state[self.current_base_station[0], self.current_base_station[1], queue, 1] = self.channels

            self.status_array[self.current_base_station[0], self.current_base_station[1], 1] += 1

            state = self.next_request(state)
            # if int(self.new_traffic[self.current_base_station[0], self.current_base_station[1]]) <= 0:
            #     state = self.next_bs(state)
            # else:
            #     state = self.next_channel(state)
        else:
            self.status_array[self.current_base_station[0], self.current_base_station[1], 1] += 1
            self.blocktimes += 1
            self.reward = -1
            # state[self.current_base_station[0], self.current_base_station[1], :, 2] -= 1
            state = self.next_request(state)
            # self.timestep +=1
        self.state = state
        # self.reward = 1 - self.get_blockprob()
        # print(state[self.current_base_station[0], self.current_base_station[1]])
        # print(self.reward)
        self.timestep +=1
        # print(state[self.current_base_station[0], self.current_base_station[1],:])
        return np.reshape(self.state, self.observation_space.shape), self.reward, self.done, {'block_prob' : self.get_blockprob(), 'timestamp' : self.get_timestamp()}

    def get_timestamp(self):
        return str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S'))

    def get_blockprob(self):
        return self.blocktimes/self.timestep

    def set_timestamp(self):
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]

    def reset(self):
        self.timestep = 1
        self.blocktimes = 0
        self.current_base_station = [0,0]
        state = np.zeros([self.row, self.col, self.channels, self.status], dtype=np.uint64)
        self.traffic_timestep = 0
        # self.traffic_timestep = np.random.randint(self.traffic_data.shape[0])
        self.temp_timestep = self.traffic_timestep
        # state[self.current_base_station[0], self.current_base_station[1], :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] / self.traffic_channel)
        self.status_array = np.zeros((self.row,self.col,2))
        for i in range(self.row):
            for j in range(self.col):
                self.status_array[i,j,0] = self.traffic_data[self.traffic_timestep, i, j, 1]
        self.bs_available = []
        for i in range(144):
            self.bs_available.append(i)
        # for i in range(self.row):
        #     for j in range(self.col):
        #         # state[i, j, :, 0] = int(self.channels * 1.5)
        #         state[i, j, :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, i, j, 1] / self.traffic_channel)
        # state[self.current_base_station[0], self.current_base_station[1], 0, 1] = self.channels + 1
        random_index = np.random.randint(len(self.bs_available))
        bs_random_index = self.bs_available[random_index]
        self.current_base_station[0] = bs_random_index // self.row
        self.current_base_station[1] = bs_random_index % self.col
        self.state = state
        return np.reshape(self.state, self.observation_space.shape)

    def render(self, mode='human'):
        class DrawText:
            def __init__(self, label:pyglet.text.Label):	
                self.label=label	
            def render(self):	
                self.label.draw()
        screen_width = 800
        screen_height = 600
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            x=30
            y=screen_height-10
            for i in range(self.row):
                x = 30 + i * 20
                for j in range(self.col):
                    bs = rendering.make_polygon([(x,y),(x-20,y-13),(x-20,y-40),(x-0,y-53),(x+20,y-40),(x+20,y-13),(x-0,y-0)], False)
                    label = pyglet.text.Label(str(int(self.channels - np.sum(self.state[i,j,:,0])/255)),
                                    font_size=10,
                                    x=x-5, y=y-25,
                                    anchor_x='left', anchor_y='center', color=(255, 0, 0, 255))
                    self.array_render[i,j] = label
                    self.viewer.add_geom(DrawText(label))
                    x = x + 40
                    self.viewer.add_geom(bs)
                y = y - 40
            self.timestamp_label = pyglet.text.Label(str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S')),
                font_size=10,
                x=screen_width-150, y=screen_height - 10,
                anchor_x='left', anchor_y='center', color=(255,0,255, 255))
            self.viewer.add_geom(DrawText(self.timestamp_label))
        else:
            for i in range(self.row):
                for j in range(self.col):
                    self.array_render[i,j].text = str(int(self.channels - np.sum(self.state[i,j,:,0])/255))
            self.timestamp_label.text = str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S'))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



        

