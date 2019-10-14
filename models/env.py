import numpy as np

class Env:
    def __init__(self, n_agent, name):
        self.name = name
        self.n_agent = n_agent
        self.state = 0
        self.base_stations = np.zeros(shape=(7,7),dtype=int)

    def reset(self):
        self.state = 0
        self.base_stations = np.zeros(shape=(7,7),dtype=int)

    def render(self):
        print("render")

    def step(self, action):
        reward = 10
        done = True
        next_state = []
        return next_state, reward, done
        

