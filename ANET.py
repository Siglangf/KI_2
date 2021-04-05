# -----------------------------------------------------EXPLANATION-----------------------------------------------------
# Learning rate, number of hidden layers, neurons per layer, activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
# -------------------------------------------------------IMPORTS-------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
# --------------------------------------------------------LOGIC--------------------------------------------------------

class ANET(nn.Module):

    def __init__(self, input_size, hidden_layers=(60, 30), lr=0.001, activation='ReLU', optimizer='Adam', EPOCHS = 3):
        super(ANET, self).__init__() # inherit from super()

        self.EPOCHS = EPOCHS
        self.input_size = input_size * input_size + 1 # size of our input, player + board size * board size
        self.lr = lr # should decay the lr ?
        self.activation_func = self.get_activation_func(activation)
        self.layers = self.get_layers(hidden_layers, self.activation_func)
        self.model = nn.Sequential(*self.layers) 
        self.optimizer = self.get_optimizer(optimizer)
        #self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = torch.nn.BCELoss(reduction="mean")

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
                "Adagrad": torch.optim.Adagrad(list(self.parameters()), lr=self.lr), #parameters are adjustable, lr specifies step size, we can decay lr
                "SGD": torch.optim.SGD(list(self.parameters()), lr=self.lr),
                "RMSprop": torch.optim.RMSprop(list(self.parameters()), lr=self.lr),
                "Adam": torch.optim.Adam(list(self.parameters()), lr=self.lr)
                }
       return optimizers[optimizer]
        
    def get_layers(self, hidden_layers, activation_function):
        """
        Generate all layers.
        """
        layers = [torch.nn.Linear(self.input_size, hidden_layers[0])] # arg: input layer size, output layer size
        layers.append(torch.nn.Dropout(p = 0.2)) #During training, randomly zeroes some of the elements of the input tensor
        layers.append(activation_function) if activation_function != None else None
        for i in range(len(hidden_layers)-1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1])) # arg: input layer size, output layer size
            layers.append(activation_function) if activation_function != None else None
        layers.append(torch.nn.Linear(hidden_layers[-1], self.input_size)) # Should output a probability distribution over all possible moves
        layers.append(torch.nn.Softmax(dim=-1))
        return layers
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Take the given state and forward it through the network. Return the output of the network.
        :param x:  tensor([2., 1., 2., 1., 0., 0., 0., 2., 1., 0., 0., 2., 0., 2., 1., 0., 1.])
        """
        with torch.no_grad():
            return self.model(x)

    def fit(self, state, target):
        """
        Train the model. The results from the MCTS is used to train our Actor Neural Network.
        : state: A game state
        : target: probability distribution, D, generated by MCTS
        By normalizing children visit counts you get a probability distribution which can be used to train the NN
        """
        state = torch.FloatTensor(state)
        target = torch.FloatTensor(target)
        for epoch in range(self.EPOCHS):
            self.optimizer.zero_grad() # gradients will contain the loss, how wrong you were
            output =  self.model(state) # the prediction
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            print(loss) # For each EPOCH the loss should get smaller
        accuracy = output.argmax(dim=0).eq(target.argmax(dim=0)).sum().numpy()/len(target)
        print(accuracy)
        return loss, accuracy #loss.item()

    
    def get_move(self, state: torch.Tensor):
        """
        :param state:  game state, state = [2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1]
        :return: probability distribution of all legal actions, and index of the best action
        """
        D = self.get_action_probabilities(state)
        action_index = torch.argmax(D).item()
        return D, action_index
   

    def get_action_probabilities(self, state: torch.Tensor):
        """
        Forward the game state through ANET and return the conditional softmax of the output.
        :param state: game state
        :return: Probability distribution over all possible actions from state.
        """
        # Forward through ANET
        output = self.forward(state)
        # Set positions that are already taken (cell filled) to zero
        mask = torch.as_tensor([int(cell == 0) for cell in state], dtype=torch.int)
        output *= mask
        # Normalize values that are not equal to zero
        sum = torch.sum(output)
        output /= sum
        return output

    
    def save_anet(self, size, level):
        torch.save(self.state_dict(), "models/{}_ANET_level_{}".format(size,level))
        print("Model has been saved to models/{}_ANET_level_{}".format(size,level))

    def load_anet(self, size, level):
        self.load_state_dict(torch.load("models/{}_ANET_level_{}".format(size,level)))
        print("Loaded model from models/{}_ANET_level_{}".format(size,level))

if __name__ == '__main__':
    #x = np.asarray([[2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1]])
    x = [2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1]
    y = [2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 0, 0, 1]
    anet = ANET(4)
    anet.fit(x,y)
    x = torch.FloatTensor(x)
    print(anet.get_move(x))
