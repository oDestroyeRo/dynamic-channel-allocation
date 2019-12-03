from keras import *
from keras.layers import *


class Policy:
    def single_channel_build_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(Conv2D(32, 1, strides=(1, 1), activation='relu', input_shape=n_inputs, padding="valid", data_format="channels_last"))
        model.add(Conv2D(64, 1, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))
        model.add(Conv2D(64, 1, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))	
        model.add(Flatten())	
        model.add(Dense(256, activation='relu'))	
        model.add(Dense(n_outputs, activation='linear', name='action'))
        model.summary()
        return model

    def multi_channel_build_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(Conv2D(32, 2, strides=(2, 2), activation='relu', input_shape=n_inputs, padding="valid", data_format="channels_last"))
        model.add(Conv2D(64, 2, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))
        model.add(Conv2D(64, 2, strides=(1, 1), activation='relu', padding="valid", input_shape=n_inputs, data_format="channels_last"))	
        model.add(Flatten())	
        model.add(Dense(256, activation='relu'))	
        model.add(Dense(n_outputs, activation='linear', name='action'))
        model.summary()
        return model

