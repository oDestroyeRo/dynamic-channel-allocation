import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
# from multi_discrete import MultiDiscrete

class SingleChannelDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.row = 10
        self.col = 10
        self.channels = 4
        self.current_base_station = [[0,0]]
        self.status = 2 # channel // location 
        self.state = np.zeros([self.row, self.col, self.status], dtype=int)

        self.action_space = spaces.Discrete(self.channels)
        self.observation_space = spaces.Box(low=0, high=self.channels + 1, shape=(self.row *self.col* self.status,), dtype=np.uint8)
        self.viewer = None
        self.seed()

        state = self.state
        for i in range(self.row):	
            for j in range(self.col):	
                # action = np.random.randint(0, self.channels)	
                # self.current_base_station[0][0] = i	
                # self.current_base_station[0][1] = j	
                # while self.check_dca(action, state) == False:	
                #     action = np.random.randint(0, self.channels)	
                state[i,j,0] = self.channels + 1
        self.temp_state = state


    
    def check_dca(self, action, state):
        c_bs_r = self.current_base_station[0][0] 
        c_bs_c = self.current_base_station[0][1]
        if c_bs_r != 0 and state[c_bs_r-1][c_bs_c][0] == action:
            return False
        if c_bs_r != self.row-1 and state[c_bs_r+1][c_bs_c][0] == action:
            return False
        if c_bs_c != 0 and state[c_bs_r][c_bs_c-1][0] == action:
            return False
        if c_bs_c != self.col-1 and state[c_bs_r][c_bs_c+1][0] == action:
            return False
        if c_bs_r != self.row-1 and c_bs_c != 0 and state[c_bs_r+1][c_bs_c-1][0] == action:
            return False
        if c_bs_r != 0 and c_bs_c != self.col-1 and state[c_bs_r-1][c_bs_c+1][0] == action:
            return False
        return True

    def next_bs(self, state):
        state[self.current_base_station[0][0]][self.current_base_station[0][1]][1] = 0
        self.current_base_station[0][1] += 1
        if self.current_base_station[0][1] >= self.col:
            self.current_base_station[0][1] = 0
            self.current_base_station[0][0] += 1
            if self.current_base_station[0][0] >= self.row:
                self.current_base_station[0][0] = 0
        state[self.current_base_station[0][0]][self.current_base_station[0][1]][1] = self.channels + 1
        return state


    def step(self, action):
        done = False
        state = self.state
        if self.check_dca(action, state):
            self.reward = 1
            state[self.current_base_station[0][0]][self.current_base_station[0][1]][0] = action

        else:
            self.reward = -1
            self.blocktimes +=1
        self.timestep +=1
        state = self.next_bs(state)
        self.state = state
        return np.reshape( state, self.observation_space.shape), self.reward, done, {'block_prob' : self.get_blockprob()}

    def get_blockprob(self):
        return self.blocktimes/self.timestep

    def reset(self):
        self.array_render = np.zeros([self.row, self.col], dtype=object)
        self.blocktimes = 0
        self.timestep = 1
        state = self.temp_state
        self.current_base_station = [[0,0]]
        state[self.current_base_station[0][0]][self.current_base_station[0][1]][1] = self.channels + 1
        self.state = state
        return np.reshape( state, self.observation_space.shape)

    def render(self, mode='human'):
        class DrawText:
            def __init__(self, label:pyglet.text.Label):	
                self.label=label	
            def render(self):	
                self.label.draw()
        screen_width = 600
        screen_height = 400
        state = self.state
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            x=30
            y=screen_height-10
            for i in range(self.row):
                x = 30 + i * 20
                for j in range(self.col):
                    bs = rendering.make_polygon([(x,y),(x-20,y-13),(x-20,y-40),(x-0,y-53),(x+20,y-40),(x+20,y-13),(x-0,y-0)], False)
                    label = pyglet.text.Label(str(int(state[i,j])),
                                    font_size=10,
                                    x=x-5, y=y-25,
                                    anchor_x='left', anchor_y='center', color=(255, 0, 0, 255))
                    self.array_render[i,j] = label
                    self.viewer.add_geom(DrawText(label))
                    x = x + 40
                    self.viewer.add_geom(bs)
                y = y - 40
        else:
            for i in range(self.row):
                for j in range(self.col):
                    self.array_render[i,j].text = str(int(state[i,j]))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



        

