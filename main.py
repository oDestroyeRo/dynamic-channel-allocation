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
    def __init__(self, state_space, action_space, episodes=850000):

        self.action_space = action_space
        self.use_nn = 0
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.99

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        n_inputs = state_space.shape
        # n_inputs = np.expand_dims(n_inputs, axis=0)
        # print(n_inputs)
        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=RMSprop(lr=0.00025,
                                            rho=0.95,
                                            epsilon=0.01),
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
        model.add(Conv2D(32, (2,2), activation='relu', input_shape=n_inputs))
        model.add(Conv2D(64, (1,1), activation='relu'))
        model.add(Conv2D(64, (1,1), activation='relu'))
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
        self.use_nn += 1
        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        return np.argmax(q_values[0])


    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)


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
    # c = Controller(args)
    # c.start()

    # root.update()ˇ

    env = gym.make('multi-agent-DCA-v0')
    # env = gym.make('Taxi-v3')
    # print(env.reset())
    # state_size = env.observation_space.shape[0]
    state_size = env.observation_space
    action_size = env.action_space


    agent = DQNAgent(env.observation_space, env.action_space)

    # should be solved in this number of episodes
    episode_count = 9999993000
    # state_size = env.observation_space
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000
    count = 0
    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # print(state)
        # state = np.reshape(state, [1, state_size])
        # state = np.eye(state_size)[state]
        state = np.expand_dims(state, 0)
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, 0)
            # next_state = np.eye(state_size)[next_state]
            # next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if count%100 == 0:
                #print(count, env.get_blockprop(), agent.epsilon, total_reward)
                with open('results/dqn_init_70_2.csv', 'a') as newFile:
                    newFileWriter = csv.writer(newFile)
                    newFileWriter.writerow([count, env.get_blockprop(), agent.epsilon, agent.use_nn])
                agent.use_nn = 0
            count += 1

        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

    # close the env and write monitor result info to disk
    env.close() 

