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
EPISODES = 200
NUM_SIMULATIONS = 2000
NUM_AGENTS = 5
BATCH_SIZE = 0.5

# ANET parameters
HIDDEN_LAYERS = (64,32) #(16,16) #[128]  # (32,32) (48,24) (16,16)
LEARNING_RATE = 0.1
ACTIVATION = 'ReLU'
OPTIMIZER = 'Adam'
EPOCHS = 1

# MCTS parameters
EPSILON = 1
EPSILON_DECAY_DEGREE = 1/2
C = 1
# Batch strategy
BATCH_TYPE_RELATIVE = True
BS_DEGREE = 3

# --------------------------------------------------------LOGIC---------------------------------------------------------


def train_anet(series_name, anet, board_size, board_actual_game, episodes, num_simulations, num_agents, batch_strategy, batch_size, eps, log=True):
    train_losses = []
    train_accuracies = []
    # CLEAR REPLAY BUFFER
    rbuf = []
    action_space = board_actual_game.get_action_space()
    anet.save_anet(series_name, board_size, 0)
    # FOR EACH EPISODE/ACTUAL GAME
    for episode in tqdm(range(1, episodes+1)):
        # INITIALIZE ACTUAL GAME BOARD to empty board
        board_actual_game.reset()
        is_final = board_actual_game.is_final()
        # INITIALIZE MCT to a single root
        board_mcts = MCTS(eps, anet, C)
        root = Node(board_actual_game)
        # FOR EACH STEP IN EPISODE/ACTUAL GAME
        while not is_final:
            board_mcts.set_root(root)
            board_mcts.run_simulations(num_simulations)
            D, new_root = board_mcts.get_probability_distribution(board_mcts.root, action_space)
            # ADD CASE TO RBUF, (F,D) where F is the feature set consisting of PID and board state, D is the target distribution
            feature = board_actual_game.get_state()

            rbuf.append((feature, D))
            #if random.random() > 0.5:
                #rbuf.append(rotated(state, D))

            action = action_space[np.argmax(D)]
            # PERFOM ACTION IN ACTUAL GAME
            _, _, is_final = board_actual_game.step(action)
            # SET NEW ROOT IN MCT
            root = new_root

        # TRAIN ANET ON RANDOM MINIBATCH 
        batch = select_batch(rbuf, batch_size, strategy = batch_strategy, deg = BS_DEGREE, batch_type_relative = BATCH_TYPE_RELATIVE)
        features = [replay[0] for replay in batch]
        labels = [replay[1] for replay in batch]
        loss, acc = anet.fit(features, labels)
        train_losses.append(loss)
        train_accuracies.append(acc)
        # DECAY EPSILON
        eps = 1-(episode/episodes)**EPSILON_DECAY_DEGREE
        # SAVE ANET
        if episode % (episodes//(num_agents-1)) == 0:
            anet.save_anet(series_name, board_size, episode)
    
    plot(train_accuracies, train_losses, board_size, num_simulations)

    # topp = TOPP(series_name=series_name, board_size=board_size, game=Hex, num_games=NUM_GAMES, episodes=episodes, num_agents=num_agents, hidden_layers=HIDDEN_LAYERS)
    # wins = topp.run_tournament()
    # log_training(series_name, board_size, episodes, num_simulations, num_agents, batch_size, loss, accuracy, wins)

def rotated(state, D):
    player = state[0]
    return (np.asarray([player] + list(state[:0:-1])), D[::-1])

def plot(train_accuracies, train_losses, board_size, simulations):
        x = np.arange(len(train_accuracies))
        fig = plt.figure(figsize=(12,5))
        title = 'Size: {}   Simulations: {} '.format(board_size, simulations)
        fig.suptitle(title, fontsize=10)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title("Accuracy")
        ax.plot(x, train_accuracies, color='tab:green', label="Train")
        plt.grid()
        plt.legend()
        ax = fig.add_subplot(gs[0,1])
        ax.set_title("Loss")
        ax.plot(x, train_losses, color='tab:orange', label="Train")
        plt.legend()
        plt.grid()
        plt.savefig("stats/size-{}".format(board_size))
        plt.close()


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
    # batch_size = len(replay_buffer)//2
    # batch = random.sample(replay_buffer, batch_size)
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
        eps = float(input("Epsilon for rollout: "))
    else:
        series_name = input("Series Name: ")
        board_size = BOARD_SIZE
        episodes = EPISODES
        num_simulations = NUM_SIMULATIONS
        num_agents = NUM_AGENTS
        batch_size = BATCH_SIZE
        eps = EPSILON

    board_actual_game = Hex(board_size)
    batch_strategy = "probability_function"

    anet = ANET(input_size=board_size, hidden_layers=HIDDEN_LAYERS,
                lr=LEARNING_RATE, activation=ACTIVATION, optimizer=OPTIMIZER, EPOCHS=EPOCHS)
 
    if os.path.exists(f"models/{series_name}_{board_size}_ANET_level_{0}"):
        # ! NOT TESTED YET
        anet.load_anet(series_name, board_size, episodes)
        series_name += "continued"
    train_anet(series_name, anet, board_size, board_actual_game, episodes, num_simulations,
               num_agents, batch_strategy, batch_size, eps)