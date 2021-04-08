# ------------------------------------------------------EXPLANATION-------------------------------------------------------
# DESCRIPTION:
# The MCTS class contains all the logic in order to perform Monte Carlo Tree Search
# Each simulation adds a single node to the game tree

# INPUT:
# - root_node: root_node for the Monte Carlo Search Tree
# -------------------------------------------------------IMPORTS-----------------------------------------------------------
import random
import numpy as np
from graph import Node
import copy
from state import State, NimState
from Simworld import Hex
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------CONSTANTS----------------------------------------------------------
WIN = 2
TIE = 1
# ----------------------------------------------------------LOGIC------------------------------------------------------------


class MCTS:
    def __init__(self, eps, anet, c=1, ANN=0):
        self.root = None
        self.anet = anet
        self.c = c
        self.eps = eps

    def set_root(self, node):
        """
        Root is the current state in "the actual game" (an episode).
        """
        self.root = node
        self.root.parent = None

    def run_simulations(self, num_simulations=4):
        if self.root.visit_count == 0:
            self.expand(self.root)

        for _ in range(num_simulations):
            # SELECT
            leaf_node = self.get_leaf_node(self.root)
            # EXPAND / ROLLOUT
            if leaf_node.visit_count > 0 and not leaf_node.state.is_final():
                self.expand(leaf_node)
                child = random.choice(leaf_node.children)
                rollout_node, reward = self.rollout(child)
            else:
                rollout_node, reward = self.rollout(leaf_node)
            # BACKUP
            self.backup(rollout_node, reward)

    def get_leaf_node(self, node):
        """
        Tree search, traversing the tree from the root node to a leaf node using the tree policy.
        """
        while node.children:
            node = self.tree_policy(node)
        return node

    def expand(self, node):
        """
        Expand the selected node, that is, add one or more children to the node
        """
        if node.state.is_final():
            return
        actions = node.state.legal_actions()
        children_nodes = []
        for action in actions:
            # make a copy of the current state
            child_state = copy.deepcopy(node.state)
            child_state.step(action)  # update the current state
            child_node = Node(child_state, action)
            children_nodes.append(child_node)
        node.add_children(children_nodes)

    def rollout(self, node, eps=0.7, ANN=None):
        """
        Play a game (rollout game) from node using the default policy.
        : return: node, reward
        """
        state = copy.deepcopy(node.state)  # make a copy of the current state
        while not state.is_final():
            action = self.default_policy(state, eps=self.eps)
            state.step(action)
        return node, state.collect_reward()

    # Backpropogation: updating relevant data on the path from the final state to the tree root node
    def backup(self, node, reward):
        node.visit_count += 1
        node.value_sum += reward
        if node.parent:
            self.backup(node.parent, reward)

    # ToDo: Should use more rollouts in the beginning and then the critic more towards the end
    def default_policy(self, state, ANN=0, eps=1, stoch=True):
        """
        : eps: During rollouts, the default policy may have a probability (ε) of choosing a random move rather than the best (so far) move.
        : state: state of the current game
        : return: action
        """
        actions = state.legal_actions()
        if random.random() < eps:
            return random.choice(actions)
        else:
            """
            ToDo: Må bruke et neural network her
            : state: [2, 1, 2, 1, 0, 0, 0, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1]
            """
            action_space = state.get_action_space()
            state = state.get_state()
            D, action_index = self.anet.get_move(state)
            return action_space[action_index]

    def tree_policy(self, node):
        """
        Tree policy: Player 1/Player 2 choose branch with the highets/lowest combination of Q(s,a) + exploration bonus, u(s,a) 
        :return: Node
        """
        children_stack = {}
        c = self.c if node.state.player == 1 else -self.c
        for child in node.children:
            children_stack[child] = child.compute_q() + child.compute_u(c)
        return max(children_stack, key=children_stack.get) if node.state.player == 1 else min(children_stack, key=children_stack.get)

    def get_probability_distribution(self, node, action_space):
        """
        From the self.root node calculate the distribution D
        """
        # Invert such that action_space maps an action to an index
        action_space = {v: k for k, v in action_space.items()}
        total = sum(child.visit_count for child in node.children)
        D = np.zeros(len(action_space))
        child_dict = {}
        for child in node.children:
            D[action_space[child.pred_action]] = child.visit_count/total
            child_dict[child] = child.visit_count/total

        return D, max(child_dict, key=child_dict.get)
