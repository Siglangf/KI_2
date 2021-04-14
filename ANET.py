# -----------------------------------------------------EXPLANATION-----------------------------------------------------
# Learning rate, number of hidden layers, neurons per layer, activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
# -------------------------------------------------------IMPORTS-------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import random
# --------------------------------------------------------LOGIC--------------------------------------------------------


class ANET(nn.Module):

    def __init__(self, input_size=None, output_size=None, hidden_layers=(60, 30, 20), lr=0.001, activation='ReLU', optimizer='Adam', EPOCHS=20, loss_function="kldiv"):
        super(ANET, self).__init__()  # inherit from super()

        self.EPOCHS = EPOCHS
        # size of our input, player + board size * board size
        self.input_size = input_size*input_size+1
        self.output_size = input_size*input_size if output_size == None else output_size
        self.lr = lr  # should decay the lr ?
        self.activation_func = self.get_activation_func(activation)
        self.layers = self.get_layers(hidden_layers, self.activation_func)
        self.model = nn.Sequential(*self.layers)
        self.optimizer = self.get_optimizer(optimizer)
        self.loss_function = self.get_loss_function(loss_function)
        self.lf = loss_function

    def get_activation_func(self, activation):
        activation_functions = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Linear": nn.Identity()
        }
        return activation_functions[activation]

    def get_optimizer(self, optimizer):
        optimizers = {
            # parameters are adjustable, lr specifies step size, we can decay lr
            "Adagrad": torch.optim.Adagrad(list(self.parameters()), lr=self.lr),
            "SGD": torch.optim.SGD(list(self.parameters()), lr=self.lr),
            "RMSprop": torch.optim.RMSprop(list(self.parameters()), lr=self.lr),
            "Adam": torch.optim.Adam(list(self.parameters()), lr=self.lr)
        }
        return optimizers[optimizer]

    def get_loss_function(self, loss_fn):
        loss_function = {
            "binary_cross_entropy": torch.nn.BCELoss(reduction="mean"),
            "mse": torch.nn.MSELoss(),
            "kldiv": torch.nn.KLDivLoss(reduction="batchmean")
        }
        return loss_function[loss_fn]

    def get_layers(self, hidden_layers, activation_function):
        """
        Generate all layers.
        """
        layers = [torch.nn.Linear(
            self.input_size, hidden_layers[0])]  # arg: (in_layer_size, out_layer_size)
        # During training, randomly zeroes some of the elements of the input tensor
        layers.append(torch.nn.Dropout(0.3))
        layers.append(activation_function)
        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(
                hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_function)
        # Should output a probability distribution over all possible moves
        layers.append(torch.nn.Linear(hidden_layers[-1], self.output_size))
        layers.append(torch.nn.Softmax(dim=-1))
        return layers

    def fit(self, states, targets):
        """
        Train the model. The results from the MCTS is used to train our Actor Neural Network.
        : state: A game state
        : target: probability distribution, D, generated by MCTS
        By normalizing children visit counts you get a probability distribution which can be used to train the NN
        """
        states = torch.FloatTensor(states)
        targets = torch.FloatTensor(targets)

        # self.model.train() # fra Kjartan....

        for epoch in range(self.EPOCHS):
            # zero the parameter gradients
            # gradients will contain the loss, how wrong you were
            self.optimizer.zero_grad()
            # forward + backward + optimize
            # the prediction, men denne inneholder ikke 0000
            outputs = self.model(states)
            if self.lf == "kldiv":
                loss = self.loss_function(F.log_softmax(outputs, -1), targets)
            else:
                loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()

            accuracy = outputs.argmax(dim=1).eq(
                targets.argmax(dim=1)).sum().numpy()/len(targets)
            print("Loss: ", loss.item(), "fake loss: ", loss)
            print("Accuracy: ", accuracy)

        # self.model(False) # fra Kjartan....

        return loss.item(), accuracy

    def get_move(self, state):
        """
        :param state:  game state, state = [2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1]
        :return: probability distribution of all legal actions, and index of the best action
        """
        state_tensor = torch.FloatTensor(state)
        D = self.get_action_probabilities(state_tensor)
        if torch.isnan(D).any().item():
            # Fix numerical issue when renormalizing
            D, action_index = self.get_random_move(state, D)
            return D, action_index
        return D.tolist(), torch.argmax(D).item()

    def get_random_move(self, state, D):
        state = state[1:]
        legal_actions = [i for i in range(len(state)) if state[i] == 0]
        D = [0 for i in range(len(D))]
        action_index = random.choice(legal_actions)
        return D, action_index

    def get_action_probabilities(self, state: torch.Tensor):
        """
        Forward the game state through ANET and return the conditional softmax of the output.
        :param state: game state
        :return: Probability distribution over all possible actions from state.
        """
        # Forward through ANET
        self.model.eval()
        with torch.no_grad():
            output = self.model(state)
        self.model.train()
        # Set positions that are already taken (cell filled) to zero
        state = state[1:]
        mask = torch.as_tensor([int(cell == 0)
                                for cell in state], dtype=torch.int)
        output *= mask
        # Normalize values that are not equal to zero
        sum = torch.sum(output)
        output /= sum
        return output

    def save_anet(self, series_name, size, level):
        torch.save(self.state_dict(),
                   "models/{}_{}_ANET_level_{}".format(series_name, size, level))
        print("Model has been saved to models/{}_{}_ANET_level_{}".format(series_name, size, level))

    def load_anet(self, series_name, size, level):
        self.load_state_dict(torch.load(
            "models/{}_{}_ANET_level_{}".format(series_name, size, level)))
        print(
            "Loaded model from models/{}_{}_ANET_level_{}".format(series_name, size, level))
