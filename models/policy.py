from keras import *
from keras.layers import *


class Policy:
    def single_channel_build_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(Dense(128, input_dim=n_inputs, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_outputs, activation='linear', name='action'))
        model.summary()
        return model

    def multi_channel_build_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(Dense(128, input_dim=n_inputs, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_outputs, activation='linear', name='action'))
        model.summary()
        return model

