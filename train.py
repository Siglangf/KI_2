# ------------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The RL algorithm involves three main activities.
# 1) Making actual moves in a game
# 2) Making simulated moves during MCTS
# 3) Updating the target policy via supervised learning
#
# -------------------------------------------------------IMPORTS--------------------------------------------------------
import random
import numpy as np
from graph import Node
from state import State, NimState
from mcts import *
from ANET import ANET
from Simworld import visualize_state, Hex
from tqdm import tqdm
import time
import math
import sys


# --------------------------------------------------------DEFAULT PARAMS---------------------------------------------------------
# Default parameters
NUMBER_OF_GAMES = 200
GAME = Hex
STARTING_PLAYER_ID = 1
SIZE = 4
EPISODES = 200
NUM_SIMULATIONS = 1000
NUM_AGENTS = 3
BATCH_SIZE = 0.3

# ANET parameters
HIDDEN_LAYERS = (60, 30, 20)
LEARNING_RATE = 0.001
ACTIVATION = 'ReLU'
OPTIMIZER = 'Adam'
EPOCHS = 20

# MCTS parameters
EPSILON = 0.7
C = 1
# Batch strategy
BS_DEGREE = 4


# --------------------------------------------------------LOGIC---------------------------------------------------------


def train_anet(series_name, anet, board_size, environment, episodes, num_simulations, num_agents, batch_strategy, batch_size):
    replay_buffer = []
    action_space = environment.get_action_space()
    anet.save_anet(series_name, board_size, 0)
    for episode in tqdm(range(1, episodes+1)):
        # Initialize environment
        environment.reset()
        is_final = environment.is_final()
        # Initialize mcts
        root = Node(environment)
        board_mcts = MCTS(root, anet, EPSILON, C)
        is_final = environment.is_final()
        while not is_final:
            board_mcts.set_root(root)
            board_mcts.run_simulations(num_simulations)
            D, new_root = board_mcts.get_probability_distribution(
                board_mcts.root, action_space)
            state = environment.get_state()
            replay_buffer.append((state, D))
            action = action_space[np.argmax(D)]
            # Perform the action in the actual environment
            state, _, is_final = environment.step(action)
            # Update root of MCST to be the choosen childstate
            root = new_root
            board_mcts.set_root(root)
        batch = select_batch(replay_buffer, batch_size,
                             strategy=batch_strategy, deg=BS_DEGREE)
        features = [replay[0] for replay in batch]
        labels = [replay[1] for replay in batch]
        anet.fit(features, labels)

        if episode % (episodes//(num_agents-1)) == 0:
            anet.save_anet(series_name, board_size, episode)


def select_batch(replay_buffer, batch_size, strategy="random", upper_percent=0.8, upper_fraq=3/4, deg=3):
    if strategy == "probability_function":
        # Choose based on probability function
        batch_size = math.ceil(batch_size*len(replay_buffer))
        probabilities = f(replay_buffer, deg)
        return random.choices(replay_buffer, weights=probabilities, k=batch_size)


def f(x, deg):
    distribution = np.zeros(len(x))
    for i in range(len(x)):
        distribution[i] = (i/len(x))**deg
    distribution /= sum(distribution)
    return distribution


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "input":
        series_name = input("Series Name: ")
        board_size = int(input("Board size: "))
        episodes = int(input("Episodes: "))
        num_simulations = int(input("Number of simulations: "))
        num_agents = int(input("Number of agents: "))
        batch_size = float(input("Batch size: "))
    else:
        series_name = input("Series Name: ")
        board_size = SIZE
        episodes = EPISODES
        num_simulations = NUM_SIMULATIONS
        num_agents = NUM_AGENTS
        batch_size = BATCH_SIZE

    environment = Hex(board_size)
    batch_strategy = "probability_function"
    anet = ANET(input_size=board_size, hidden_layers=HIDDEN_LAYERS,
                lr=LEARNING_RATE, activation=ACTIVATION, optimizer=OPTIMIZER, EPOCHS=EPOCHS)
    train_anet(series_name, anet, board_size, environment, episodes, num_simulations,
               num_agents, batch_strategy, batch_size)
