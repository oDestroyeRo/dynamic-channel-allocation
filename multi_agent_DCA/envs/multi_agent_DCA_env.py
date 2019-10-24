import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
# from multi_discrete import MultiDiscrete

class MultiAgentDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        self.row = 7
        self.col = 7
        self.channels = 70
        # self.global_base_stations = np.empty([self.row, self.col], dtype=int)
        self.current_base_station = np.random.randint(self.col, size=(1, 2))
        self.reward = 0
        self.timestep = 0
        self.state = np.empty([self.row, self.col, self.channels], dtype=int)
        for i in range(self.row):
            for j in range(self.col):
                action = np.random.randint(0, self.channels)
                self.current_base_station[0][0] = i
                self.current_base_station[0][1] = j
                while self.check_dca(action) == False:
                    action = np.random.randint(0, self.channels)
                self.state[i][j][action] = 1

        # for i in range(self.row):
        #     for j in range(self.col):
        #         self.state[i][j] = -1

            

        # self.agents = 49
        # self.action_space = []
        # self.observation_space = []
        # self.action_space.append(spaces.Discrete(2))
        # self.observation_space.append(spaces.Discrete(70))
        # agent.action.c = np.zeros(70)
        self.action_space = spaces.Discrete(self.channels)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.row ,self.col, self.channels), dtype=np.uint8)
        # self.observation_space = spaces.Discrete(self.row * self.col * self.channels)
        # self.observation_space = spaces.Box(low=0, high=70, shape=(7, 70), dtype=np.uint8)
        self.viewer = None
        self.seed()

        self.duptimes = 0
        # self.min_position = -1.2
        # self.max_position = 0.6
        # self.max_speed = 0.07
        # self.low = np.array([self.min_position, -self.max_speed])
        # self.high = np.array([self.max_position, self.max_speed])
        # print(np.array(spaces.Box(self.low, self.high, dtype=np.float32)))
    
    def check_dca(self, action):
        c_bs_r = self.current_base_station[0][0] 
        c_bs_c = self.current_base_station[0][1]
 
        # if c_bs_r != 0 and action == self.state[c_bs_r-1][c_bs_c]:
        #     return False
        # if c_bs_r != self.row-1 and action == self.state[c_bs_r+1][c_bs_c]:
        #     return False
        # if c_bs_c != 0 and action == self.state[c_bs_r][c_bs_c-1]:
        #     return False
        # if c_bs_c != self.col-1 and action == self.state[c_bs_r][c_bs_c+1]:
        #     return False
        # if c_bs_r != self.row-1 and c_bs_c != 0 and action == self.state[c_bs_r+1][c_bs_c-1]:
        #     return False
        # if c_bs_r != 0 and c_bs_c != self.col-1 and action == self.state[c_bs_r-1][c_bs_c+1]:
        #     return False
        # return True

        if c_bs_r != 0 and self.state[c_bs_r-1][c_bs_c][action] == 1:
            return False
        if c_bs_r != self.row-1 and self.state[c_bs_r+1][c_bs_c][action] == 1:
            return False
        if c_bs_c != 0 and self.state[c_bs_r][c_bs_c-1][action] == 1:
            return False
        if c_bs_c != self.col-1 and self.state[c_bs_r][c_bs_c+1][action] == 1:
            return False
        if c_bs_r != self.row-1 and c_bs_c != 0 and self.state[c_bs_r+1][c_bs_c-1][action] == 1:
            return False
        if c_bs_r != 0 and c_bs_c != self.col-1 and self.state[c_bs_r-1][c_bs_c+1][action] == 1:
            return False
        return True

    def step(self, action):
        if self.check_dca(action):
            self.reward = 1.0 - (0.001 * self.duptimes)
            if (self.duptimes > 0):
                print(action, self.duptimes)
            self.state[self.current_base_station[0][0]][self.current_base_station[0][1]] = 0
            self.state[self.current_base_station[0][0]][self.current_base_station[0][1]][action] = 1
            self.current_base_station = np.random.randint(self.col, size=(1, 2))
            self.duptimes = 0
        else:
            # self.reward = 1.0 
            self.reward = 0 - (0.001 * self.duptimes)
            self.blocktimes +=1
            self.duptimes += 1
        self.timestep +=1
        # for agent in actions:

        #     if agent.action == 1:
        # channel = self.state
        return self.state, self.reward, False, {}

    def get_blockprop(self):
        return self.blocktimes/self.timestep


    def reset(self):
        self.timestep = 0
        self.blocktimes = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        # carty = 100 # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # self.track = rendering.FilledPolygon([(0,0), (100,100), (200,200), (300,300)])
            # self.track.add_attr(rendering.Transform(translation=(0, 10)))
            # # self.track.set_color(222,222,222)
            # self.viewer.add_geom(self.track)

            x=30
            y=screen_height-10

            for i in range(self.row):
                x = 30 + i * 20
                for j in range(self.col):
                    bs = rendering.make_polygon([(x,y),(x-20,y-13),(x-20,y-40),(x-0,y-53),(x+20,y-40),(x+20,y-13),(x-0,y-0)], False)

                    x = x + 40
                    self.viewer.add_geom(bs)
                y = y - 40
            label = pyglet.text.Label('Hello, world',
                                    font_name='Times New Roman',
                                    font_size=36,
                                    x=20, y=20,
                                    anchor_x='center', anchor_y='center')
            label.draw()
            # self.viewer.add_geom(DrawText(label))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# class DrawText:
#     def __init__(self, label:pyglet.text.Label):
#         self.label=label
#     def render(self):
#         self.label.draw()