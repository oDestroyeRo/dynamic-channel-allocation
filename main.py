import os
from keras import *
from keras.layers import *
from keras.optimizers import *

import argparse
import multi_agent_DCA
import gym

from collections import deque
import numpy as np
import random
import csv

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,
                              2,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=state_size,
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              2,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=state_size,
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              2,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=state_size,
                              data_format="channels_last"))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(action_size))
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=self.learning_rate),
                           metrics=["accuracy"])
        self.model.summary()
        return self.model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # args = parse_args()
    # c = Controller(args)
    # c.start()

    # root.update()ˇ

    env = gym.make('multi-agent-DCA-v0')
    # state_size = env.observation_space.shape[0]
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 64

    state = env.reset()
    # print(state.shape())
    # state = np.reshape(state, [1, state_size])
    state = np.expand_dims(state, axis=0)
    for time in range(99999999):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if time%100 == 0:
            print(time, env.get_blockprop())
            with open('results/dqn_init_70.csv', 'a') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([time, env.get_blockprop()])

        # next_state = np.reshape(next_state, [1, state_size])
        next_state = np.expand_dims(next_state, axis=0)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

