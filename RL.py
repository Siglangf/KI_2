# ------------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The RL algorithm involves three main activities.
# 1) Making actual moves in a game
# 2) Making simulated moves during MCTS
# 3) Updating the target policy via supervised learning
#
# -------------------------------------------------------IMPORTS--------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# --------------------------------------------------------LOGIC---------------------------------------------------------

class RL:

    def __init__(self, ANN, mcts):
        self.ANN = ANN
        self.mcts = mcts