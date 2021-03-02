import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class CNN:

    def __init__(self, lr, input_layer, hidden_layers, output_layer):
        self.lr = lr #learning rate
        self.input_layer  = input_layer
        self.hidden_layers = hidden_layers

    def generate_NN(self):
        model = Sequential()
        model.add(Dense(self.input_layer, activation='relu', input_dim=self.input_layer)) #input layer, activation relu: just positive numbers as weights and output
        for layer in self.hidden_layers:
            model.add(Dense(layer, activation='relu')) #hidden layers
        model.add(Dense(1)) #one output layer, e.i. produces ONE expected value for a state
        optimizer = tf.optimizers.Adagrad(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), run_eagerly = True) #hyper parameters which are subject to changes, fine tuning
        return model

    # Target: value from a rollout?
    def fit(self):
        return