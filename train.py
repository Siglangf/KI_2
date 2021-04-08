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
NUMBER_OF_GAMES = 150
GAME = Hex
STARTING_PLAYER_ID = 1
SIZE = 4
NUMBER_SEARCH_GAMES = 1000
NUM_AGENTS = 3
BATCH_SIZE = 40
# --------------------------------------------------------LOGIC---------------------------------------------------------


def train_anet(series_name, anet, board_size, environment, episodes, num_simulations, num_agents, batch_strategy, batch_size):
    replay_buffer = []
    action_space = environment.get_action_space()
    for episode in tqdm(range(episodes+1)):
        # Initialize environment
        environment.reset()
        is_final = environment.is_final()
        # Initialize mcts
        root = Node(environment)
        board_mcts = MCTS(root, anet)
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
                             strategy=batch_strategy)
        features = [replay[0] for replay in batch]
        labels = [replay[1] for replay in batch]
        anet.fit(features, labels)

        if episode % (episodes//(num_agents-1)) == 0:
            anet.save_anet(series_name, board_size, episode)


def select_batch(replay_buffer, batch_size, strategy="random", upper_percent=0.8, upper_fraq=3/4, deg=3):
    if strategy == "random":
        """
        Choose random samples from the replay buffer
        """
        while batch_size > len(replay_buffer):
            batch_size = batch_size//2
        return random.sample(replay_buffer, batch_size)
    if strategy == "mixed":
        """
        Split rbuf in two partitions,
        and draw a random sample with from each partition, 
        with the number drawn in each partition is determined by a percentage
        """
        split = math.ceil(len(replay_buffer)*upper_fraq)
        lower = replay_buffer[:split]
        upper = replay_buffer[split:]
        while batch_size*upper_percent > len(upper):
            batch_size = batch_size//2
        num_upper_batch = math.floor(batch_size*upper_percent)
        num_lower_batch = batch_size-num_upper_batch
        upper_batch = random.sample(upper, num_upper_batch)
        lower_batch = random.sample(lower, num_lower_batch)
        return lower_batch+upper_batch
    if strategy == "probability_function":
        # Choose based on probability function
        while batch_size > len(replay_buffer):
            batch_size = batch_size//2
        probabilities = f(replay_buffer, deg)
        return random.choices(replay_buffer, weights=probabilities, k=batch_size)


def f(x, deg):
    distribution = np.zeros(len(x))
    for i in range(len(x)):
        distribution[i] = (i/len(x))**deg
    distribution /= sum(distribution)
    return distribution


if __name__ == '__main__':
    if sys.argv[1] == "input":
        series_name = input("Series Name: ")
        board_size = int(input("Board size: "))
        episodes = int(input("Episodes: "))
        num_simulations = int(input("Number of simulations: "))
        num_agents = int(input("Number of agents: "))
        batch_size = int(input("Batch size: "))
    else:
        series_name = sys.argv[1]
        board_size = int(sys.argv[2])
        episodes = int(sys.argv[3])
        num_simulations = int(sys.argv[4])
        num_agents = int(sys.argv[5])
        batch_size = int(sys.argv[6])
    environment = Hex(board_size)
    batch_strategy = "probability_function"
    anet = ANET(board_size)
    train_anet(series_name, anet, board_size, environment, episodes, num_simulations,
               num_agents, batch_strategy, batch_size)
