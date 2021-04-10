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
import os
from TOPP import TOPP


# --------------------------------------------------------DEFAULT PARAMS---------------------------------------------------------

# Default parameters
GAME = Hex
STARTING_PLAYER_ID = 1
BOARD_SIZE = 4
EPISODES = 50
NUM_SIMULATIONS = 1000
NUM_AGENTS = 5
BATCH_SIZE = 100

# ANET parameters
HIDDEN_LAYERS = (64, 32)
LEARNING_RATE = 0.01
ACTIVATION = 'ReLU'
OPTIMIZER = 'Adam'
EPOCHS = 20

# MCTS parameters
EPSILON = 0.4
C = 1.5
# Batch strategy
BATCH_TYPE_RELATIVE = False
BS_DEGREE = 5

# Topp parameters
NUM_GAMES = 100

# --------------------------------------------------------LOGIC---------------------------------------------------------


def train_anet(series_name, anet, board_size, environment, episodes, num_simulations, num_agents, batch_strategy, batch_size, log=True):
    replay_buffer = []
    action_space = environment.get_action_space()
    anet.save_anet(series_name, board_size, 0)
    for episode in tqdm(range(1, episodes+1)):
        # Initialize environment
        environment.reset()
        is_final = environment.is_final()
        # Initialize mcts
        root = Node(environment)
        board_mcts = MCTS(EPSILON, anet, C)
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
                             strategy=batch_strategy, deg=BS_DEGREE, batch_type_relative=BATCH_TYPE_RELATIVE)
        features = [replay[0] for replay in batch]
        labels = [replay[1] for replay in batch]
        loss, accuracy = anet.fit(features, labels)

        if episode % (episodes//(num_agents-1)) == 0:
            anet.save_anet(series_name, board_size, episode)

    topp = TOPP(series_name=series_name, board_size=board_size, game=Hex,
                num_games=NUM_GAMES, episodes=episodes, num_agents=num_agents, hidden_layers=HIDDEN_LAYERS)
    wins = topp.run_tournament()
    log_training(series_name, board_size, episodes, num_simulations,
                 num_agents, batch_size, loss, accuracy, wins)


def log_training(series_name, board_size, episodes, num_simulations, num_agents, batch_size, loss, accuracy, wins=[]):
    total = sum(wins)
    wins = [number/total for number in wins]
    winstring = "\n".join(
        [f"Win {i}: {score}" for i, score in enumerate(wins)])
    with open(f"stats/log.txt", 'a') as f:
        stats = f"\n\
############################################ {series_name} #############################################\n\
# Default parameters\n\
BOARD_SIZE ={board_size}\n\
EPISODES = {episodes}\n\
NUM_SIMULATIONS = {num_simulations}\n\
NUM_AGENTS =  {num_agents}\n\
BATCH_SIZE = {batch_size}\n\n\
# ANET parameters\n\
HIDDEN_LAYERS = {HIDDEN_LAYERS}\n\
LEARNING_RATE = {LEARNING_RATE}\n\
ACTIVATION = {ACTIVATION}\n\
OPTIMIZER = {OPTIMIZER}\n\
EPOCHS = {EPOCHS}\n\n\
# MCTS parameters\n\
EPSILON = {EPSILON}\n\
C = {C}\n\
# Batch strategy\n\
BS_DEGREE = {BS_DEGREE}\n\n\
# Results \n\
Loss: {loss}\n\
Accuracy: {accuracy}\n\
# Win percentages of agents\n\
 {winstring}"
        f. write(stats)


def select_batch(replay_buffer, batch_size, strategy="probability_function",  deg=3, batch_type_relative=True):
    if batch_type_relative:
        if batch_size > 1 or batch_size < 0:
            raise ValueError(
                "If batchsize is relative it must be number between 0 and 1.")
        batch_size = math.ceil(batch_size*len(replay_buffer))
    elif batch_size > len(replay_buffer):
        return replay_buffer
    if strategy == "probability_function":
        # Choose based on probability function
        probabilities = f(replay_buffer, deg)
        batch_idx = np.random.choice(
            [i for i in range(len(replay_buffer))], batch_size, p=probabilities)
        batch = [replay_buffer[i] for i in batch_idx]
        return batch


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
        board_size = BOARD_SIZE
        episodes = EPISODES
        num_simulations = NUM_SIMULATIONS
        num_agents = NUM_AGENTS
        batch_size = BATCH_SIZE

    environment = Hex(board_size)
    batch_strategy = "probability_function"

    anet = ANET(input_size=board_size, hidden_layers=HIDDEN_LAYERS,
                lr=LEARNING_RATE, activation=ACTIVATION, optimizer=OPTIMIZER, EPOCHS=EPOCHS)
    if os.path.exists(f"models/{series_name}_{board_size}_ANET_level_{0}"):
        # ! NOT TESTED YET
        anet.load_anet(series_name, board_size, episodes)
        series_name += "continued"
    train_anet(series_name, anet, board_size, environment, episodes, num_simulations,
               num_agents, batch_strategy, batch_size)
