import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
import math
import pyglet
# from multi_discrete import MultiDiscrete

class MultiChannelDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
        self.row = 10
        self.col = 10
        # self.channels = 150
        self.traffic_channel = 1000
        self.channels = np.max(self.traffic_data[:,:,:,1]) / self.traffic_channel
        self.channels = math.ceil(self.channels)
        self.channels = int(self.channels) // 2

        self.global_base_stations = np.zeros([self.row ,self.col ,self.channels], dtype=np.uint8)
        # self.current_base_station = np.random.randint(self.col, size=(1, 2))
        self.current_base_station = np.array([[0,0]])
        # self.temp_cbs = self.current_base_station
        self.reward = 0
        self.timestep = 1
        self.blocktimes = 0
        self.state = None
        self.traffic_timestep = 0
        self.next_channel = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0][0], self.current_base_station[0][1], 1] / self.traffic_channel)
        self.remain_channel = self.next_channel
        # self.temp_nc = self.next_channel
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]

        # self.temp_gbs = self.global_base_stations 
        self.action_space = spaces.Discrete(self.channels)
        # self.observation_space = spaces.Discrete(self.row * self.col)
        self.position = 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row *self.col *self.channels, ), dtype=np.uint8)

        self.viewer = None
        self.seed()

        self.array_render = np.zeros([self.row, self.col], dtype=object)


        # self.interation = 0


    
    def check_dca(self, action):
        c_bs_r = self.current_base_station[0][0] 
        c_bs_c = self.current_base_station[0][1]

        if self.global_base_stations[c_bs_r][c_bs_c][action] == 1:
            # print(1)
            # print(self.current_base_station)
            return False
        if c_bs_r != 0 and self.global_base_stations[c_bs_r-1][c_bs_c][action] == 1:
            # print(2)
            # print(self.current_base_station)
            return False
        if c_bs_r != self.row-1 and self.global_base_stations[c_bs_r+1][c_bs_c][action] == 1:
            # print(3)
            # print(self.current_base_station)
            return False
        if c_bs_c != 0 and self.global_base_stations[c_bs_r][c_bs_c-1][action] == 1:
            # print(4)
            # print(self.current_base_station)
            return False
        if c_bs_c != self.col-1 and self.global_base_stations[c_bs_r][c_bs_c+1][action] == 1:
            # print(5)
            # print(self.current_base_station)
            return False
        if c_bs_r != self.row-1 and c_bs_c != 0 and self.global_base_stations[c_bs_r+1][c_bs_c-1][action] == 1:
            # print(6)
            # print(self.current_base_station)
            return False
        if c_bs_r != 0 and c_bs_c != self.col-1 and self.global_base_stations[c_bs_r-1][c_bs_c+1][action] == 1:
            # print(7)
            # print(self.current_base_station)
            return False
        return True

    def check_channel_avalable(self):
        c_bs_r = self.current_base_station[0][0] 
        c_bs_c = self.current_base_station[0][1]

        used_channel = np.zeros((self.channels,), dtype=int)
        result = np.array(np.where(self.global_base_stations[c_bs_r][c_bs_c] == 1))[0,:]
        if c_bs_r != 0:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r-1][c_bs_c] == 1))[0,:],result)
        if c_bs_r != self.row-1:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r+1][c_bs_c] == 1))[0,:],result)
        if c_bs_c != 0:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r][c_bs_c-1] == 1))[0,:],result)
        if c_bs_c != self.col-1:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r][c_bs_c+1] == 1))[0,:],result)
        if c_bs_r != self.row-1 and c_bs_c != 0:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r+1][c_bs_c-1] == 1))[0,:],result)
        if c_bs_r != 0 and c_bs_c != self.col-1:
            result = np.append(np.array(np.where(self.global_base_stations[c_bs_r-1][c_bs_c+1] == 1))[0,:],result)
        used_channel[np.unique(result)] = 1
        if np.sum(used_channel) >= self.channels:
            # print(np.sum(used_channel))
            return False
        return True

    def step(self, action):
        self.done = False
        if self.check_dca(action):
            self.reward = 0
            self.remain_channel -= 1
            # self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]] = 0
            self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]][action] = 1
            if self.remain_channel == 0:
                self.reward = +1
                # self.current_base_station[0][0] += 1
                self.next_bs()
            # else:
            #     self.reward = -1 
            # self.timestep +=1
                
            # self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]] = 2
            # done = False
            # print(self.timestep)
            # print(self.check_channel_avalable())
            # print(self.channels)
        else:
            # self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]][action] = 2
            # self.timestep +=1
            if not self.check_channel_avalable():
                self.next_bs()
                self.reward = -1
            else:
                self.reward = -10
                self.blocktimes +=1
                self.done = False
            # self.remain_channel -= 1
            # self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]][action] = 0
            # if self.remain_channel == 0:
            #     # self.current_base_station[0][0] += 1
            #     self.current_base_station[0][1] += 1
            #     if self.current_base_station[0][1] >= self.col:
            #         self.current_base_station[0][0] += 1
            #         if self.current_base_station[0][0] >= self.row and self.current_base_station[0][1] >= self.col:
            #             self.traffic_timestep += 1
            #             self.get_timestamp()
            #             if self.traffic_timestep >= self.traffic_data.shape[0]:
            #                 self.reward = 0.0
            #                 done = True
            #         self.current_base_station[0][1] = 0
            #         if self.current_base_station[0][0] >= self.row:
            #             self.current_base_station[0][0] = 0
            #             self.global_base_stations = np.zeros([self.row, self.col, self.channels], dtype=int)
            #     self.next_channel = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0][0], self.current_base_station[0][1], 1] / self.traffic_channel)
            #     self.remain_channel = self.next_channel
        # self.interation += 1
        print(self.timestep, action, self.check_channel_avalable(), self.current_base_station, self.blocktimes)
        self.timestep +=1
        if self.timestep >= 1000:
            self.done = True
            self.reward = 0
            # print(self.get_blockprob())
        self.state = self.global_base_stations

        self.state = np.reshape(self.state, (self.row * self.col * self.channels, ))
        # self.state = np.append(self.state, self.encode(self.current_base_station[0,0], self.current_base_station[0,1]))
        return self.state, self.reward, self.done, {'blockprob' : self.get_blockprob()}

    def next_bs(self):
        self.current_base_station[0][1] += 1
        if self.current_base_station[0][1] >= self.col:
            self.current_base_station[0][0] += 1
            if self.current_base_station[0][0] >= self.row and self.current_base_station[0][1] >= self.col:
                self.traffic_timestep += 1
                self.get_timestamp()
                if self.traffic_timestep >= self.traffic_data.shape[0]:
                    self.reward = 0.0
                    self.done = True
            self.current_base_station[0][1] = 0
            if self.current_base_station[0][0] >= self.row:
                self.current_base_station[0][0] = 0
                self.global_base_stations = np.zeros([self.row, self.col, self.channels], dtype=int)
        self.next_channel = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0][0], self.current_base_station[0][1], 1] / self.traffic_channel)
        self.remain_channel = self.next_channel
        self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]] = 0

    def get_blockprob(self):
        return self.blocktimes/self.timestep

    def get_timestamp(self):
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]
        return self.timestamp

    def reset(self):
        # print("reset")
        self.timestep = 1
        self.blocktimes = 0
        # self.interation = 0
        # self.global_base_stations = self.temp_gbs
        self.current_base_station = np.array([[0,0]])
        # self.current_base_station = np.random.randint(self.col, size=(1, 2))
        self.global_base_stations = np.zeros([self.row, self.col, self.channels], dtype=np.uint8)
        # self.next_channel = self.temp_nc
        self.reward = 0
        self.next_channel = math.ceil(self.traffic_data[ self.traffic_timestep, self.current_base_station[0][0], self.current_base_station[0][1], 1] / self.traffic_channel)
        self.remain_channel = self.next_channel
        self.traffic_timestep = 0
        # self.current_base_station = self.temp_cbs
        # self.global_base_stations[self.current_base_station[0][0]][self.current_base_station[0][1]] = 2
        self.state = self.global_base_stations
        self.state = np.reshape(self.state, (self.row * self.col * self.channels, ))
        # self.state = np.append(self.state, self.encode(self.current_base_station[0,0], self.current_base_station[0,1]))
        return self.state

    def render(self, mode='human'):
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
                    label = pyglet.text.Label(str(np.sum(self.global_base_stations[i,j,:])),
                                    font_size=10,
                                    x=x-5, y=y-25,
                                    anchor_x='left', anchor_y='center', color=(255, 0, 0, 255))
                    self.array_render[i,j] = label
                    self.viewer.add_geom(DrawText(label))
                    x = x + 40
                    self.viewer.add_geom(bs)
                y = y - 40
            self.timestamp_label = pyglet.text.Label(str(datetime.utcfromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')),
                font_size=10,
                x=screen_width-150, y=screen_height - 10,
                anchor_x='left', anchor_y='center', color=(255,0,255, 255))
            self.viewer.add_geom(DrawText(self.timestamp_label))
        else:
            for i in range(self.row):
                for j in range(self.col):
                    self.array_render[i,j].text = str(np.sum(self.global_base_stations[i,j,:]))
            self.timestamp_label.text = str(datetime.utcfromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S'))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



        

