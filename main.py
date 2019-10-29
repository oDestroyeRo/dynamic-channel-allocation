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

class DQNAgent():
    def __init__(self, state_space, action_space, episodes=100000):

        self.action_space = action_space
        # experience buffer
        self.memory = []
        self.memory_size = 900000
        # discount rate
        self.gamma = 0.95

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.005
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))


        # Q Network for training
        n_inputs = state_space.shape

        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam(),
                           metrics=["accuracy"])
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        self.ddqn = True
        if self.ddqn:
            print("----------Double DQN--------")
        else:
            print("-------------DQN------------")

    
    # Q Network is 256-256-256 MLP
    def build_model(self, n_inputs, n_outputs):
        model = Sequential()
        # model.add(Embedding(n_inputs, 10, input_length=1))
        # model.add(Reshape((10,)))
        model.add(Conv2D(32, 2, strides=(1, 1), activation='relu', input_shape=n_inputs, padding="valid", data_format="channels_last"))
        model.add(Conv2D(64, 2, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))
        model.add(Conv2D(64, 2, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(n_outputs, activation='linear', name='action'))
        model.summary()
        return model


    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)


    # copy trained Q Network params to target Q Network
    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())


    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # explore .- do random action
            return self.action_space.sample()
        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        return np.argmax(q_values[0])


    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)


    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def get_target_q_value(self, next_state):
        # max Q value among next state's actions
        if self.ddqn:
            # DDQN
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the max Q value among next actions
            # selection and evaluation of action is on the target Q Network
            # Q_max = max_a' Q_target(s', a')
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            # state = np.reshape(state,)
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    
    # decrease the exploration, increase exploitation
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # args = parse_args()

    env = gym.make('multi-agent-DCA-v0')

    state_size = env.observation_space
    action_size = env.action_space


    agent = DQNAgent(env.observation_space, env.action_space)

    # should be solved in this number of episodes
    episode_count = 1000000

    batch_size = 64

    count = 0
    total_reward = 0
    total_block_prob = 0
    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()

        state = np.expand_dims(state, 0)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, 0)

            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            count += 1
        total_block_prob += env.get_blockprob()
        if episode%100 == 0:
            #print(count, env.get_blockprop(), agent.epsilon, total_reward)
            with open('results/dqn_init_70_ran_3.csv', 'a') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([count, total_block_prob/100, total_reward/100, agent.epsilon])
            total_reward = 0
            total_block_prob = 0
            env.blocktimes = 0
            env.timestep = 0
        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)


    # close the env and write monitor result info to disk
    env.close() 

