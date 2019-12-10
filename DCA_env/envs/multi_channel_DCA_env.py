import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
import math
from datetime import datetime
from pytz import timezone
import pytz
# from multi_discrete import MultiDiscrete

la = timezone("CET")


class MultiChannelDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
        self.row = 10
        self.col = 10
        self.traffic_channel = 1000
        self.channels = np.max(self.traffic_data[:,:,:,1]) / self.traffic_channel
        self.channels = math.ceil(self.channels)
        self.channels = int(self.channels)
        self.status = 3 #channel available //location // remain
        self.current_base_station = np.array([[0,0]])
        self.reward = 0
        self.timestep = 1
        self.blocktimes = 0
        self.state = None
        self.traffic_timestep = 0
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]

        self.action_space = spaces.Discrete(self.channels)
        self.observation_space = spaces.Box(low=-1, high=255, shape=(self.row *self.col *self.channels *self.status,), dtype=np.uint8)

        self.viewer = None
        self.seed()

        self.array_render = np.zeros([self.row, self.col], dtype=object)



    
    def check_dca(self, action, state):
        c_bs_r = self.current_base_station[0, 0]
        c_bs_c = self.current_base_station[0, 1]
        if state[c_bs_r, c_bs_c, action, 0] == 255:
            return False
        if c_bs_r != 0 and state[c_bs_r-1, c_bs_c, action, 0] == 255:
            return False
        if c_bs_r != self.row-1 and state[c_bs_r+1, c_bs_c, action, 0] == 255:
            return False
        if c_bs_c != 0 and state[c_bs_r, c_bs_c-1, action, 0] == 255:
            return False
        if c_bs_c != self.col-1 and state[c_bs_r, c_bs_c+1, action, 0] == 255:
            return False
        if c_bs_r != self.row-1 and c_bs_c != 0 and state[c_bs_r+1, c_bs_c-1, action, 0] == 255:
            return False
        if c_bs_r != 0 and c_bs_c != self.col-1 and state[c_bs_r-1, c_bs_c+1, action, 0] == 255:
            return False
        return True

    def check_channel_avalable(self, state):
        c_bs_r = self.current_base_station[0, 0] 
        c_bs_c = self.current_base_station[0, 1]

        used_channel = np.zeros((self.channels,), dtype=int)
        result = np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:]
        if c_bs_r != 0:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        if c_bs_r != self.row-1:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        if c_bs_c != 0:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        if c_bs_c != self.col-1:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        if c_bs_r != self.row-1 and c_bs_c != 0:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        if c_bs_r != 0 and c_bs_c != self.col-1:
            result = np.append(np.array(np.where(state[c_bs_r, c_bs_c, :, 0] == 255))[0,:],result)
        used_channel[np.unique(result)] = 1
        if np.sum(used_channel) >= self.channels:
            return False
        return True

    def next_bs(self, state):
        # state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 2] = 0
        state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 1] = 0
        self.current_base_station[0, 1] += 1
        if self.current_base_station[0, 1] >= self.col:
            self.current_base_station[0, 0] += 1
            if self.current_base_station[0, 0] >= self.row and self.current_base_station[0, 1] >= self.col:
                # print("change")
                # self.traffic_timestep += 1
                # self.set_timestamp()
                state = np.zeros([self.row, self.col, self.channels, self.status], dtype=np.uint8)
                for i in range(self.row):
                    for j in range(self.col):
                        state[i, j, :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, i, j, 1] / self.traffic_channel)
                if self.traffic_timestep >= self.traffic_data.shape[0]:
                    self.reward = 0
                    self.done = True
            self.current_base_station[0, 1] = 0
            if self.current_base_station[0, 0] >= self.row:
                self.current_base_station[0, 0] = 0
        # state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0, 0], self.current_base_station[0, 1], 1] / self.traffic_channel)
        state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 1] = 255
        state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 0] = 0
        return state

    def step(self, action):
        self.done = False
        state = self.state
        if not self.check_channel_avalable(state):
            # self.reward = -int(state[self.current_base_station[0, 0], self.current_base_station[0, 1], action, 2])
            self.reward = -1
            self.blocktimes += 1
            state = self.next_bs(state)
            self.timestep +=1

        elif self.check_dca(action, state):
            self.reward = 0
            state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 2] -= 1
            state[self.current_base_station[0, 0], self.current_base_station[0, 1], action, 0] = 255
            if int(state[self.current_base_station[0, 0], self.current_base_station[0, 1], action, 2]) <= 0:
                self.reward = 1
                state = self.next_bs(state)
                self.timestep +=1
        else:
            # self.reward = -1
            self.blocktimes += 1
            state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 0] = 0
            # self.reward = -int(state[self.current_base_station[0, 0], self.current_base_station[0, 1], action, 2])
            self.reward = -1
            # state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 2] -= 1
            state = self.next_bs(state)
            self.timestep +=1
        self.state = state
        return np.reshape(self.state, self.observation_space.shape), self.reward, self.done, {'block_prob' : self.get_blockprob()}


    def get_blockprob(self):
        return self.blocktimes/self.timestep

    def set_timestamp(self):
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]

    def reset(self):
        self.timestep = 1
        self.blocktimes = 0
        self.current_base_station = np.array([[0,0]])
        state = np.zeros([self.row, self.col, self.channels, self.status], dtype=np.uint8)
        state[:, :, :, 0] = 255
        self.traffic_timestep = 0
        # state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0, 0], self.current_base_station[0, 1], 1] / self.traffic_channel)
        for i in range(self.row):
            for j in range(self.col):
                state[i, j, :, 2] = math.ceil(self.traffic_data[ self.traffic_timestep, i, j, 1] / self.traffic_channel)
        state[self.current_base_station[0, 0], self.current_base_station[0, 1], :, 1] = 255
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
                    label = pyglet.text.Label(str(int(self.channels - np.sum(self.global_base_stations[i,j,:]))),
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
                    self.array_render[i,j].text = str(int(self.channels - np.sum(self.global_base_stations[i,j,:])))
            self.timestamp_label.text = str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S'))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



        

