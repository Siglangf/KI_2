# ------------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The MCTS class contains all the logic in order to perform Monte Carlo Tree Search

# INPUT:
# - root_node: root_node for the Monte Carlo Search Tree
# -------------------------------------------------------IMPORTS--------------------------------------------------------
import random
import numpy as np
from graph import Node
import copy
from state import State, NimState
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
# --------------------------------------------------------LOGIC---------------------------------------------------------


class MCTS:
    def __init__(self, game, root, c=1, ANN=0):
        self.game = game
        self.root = root
        self.c = c

    def get_probability_distribution(self, node):
        total = sum(child.visit_count for child in node.children)
        D = {}
        for child in node.children:
            D[child] = child.visit_count/total
        return D

    # Tree search: traversing the tree from the root node to a leaf node using the tree policy.
    # We start at the root and find a path to a frontier node by iteratively selecting the best child.
    def get_leaf_node(self):
        node = self.root
        while node.children:
            node = self.tree_policy(node)
        return node

    # Node expansion: Expand the selected node, that is, add one or more children to the node (usually only one).
    # Kankje ikke generere alle children til en node?
    def expand_node(self, node):
        if node.state.is_game_over():
            return node
        moves = node.state.get_legal_moves()
        children_nodes = []
        for move in moves:
            # make a copy of the current state
            child_state = copy.copy(node.state)
            child_state.do_move(move)  # update the current state
            child_node = Node(child_state)
            children_nodes.append(child_node)
        node.add_children(children_nodes)

    # Leaf evaluation: Return 1 if player win, 0 if tie, else return -1
    def get_leaf_evalutation(self, node):
        # Do a rollout from child node, aka. Play a random game from on of the generated child nodes.
        if node.children:
            node = random.choice(node.children)
        winner = self.rollout(node)
        if winner:  # player 1 win
            return node, 1
        elif not winner:  # player 2 win
            return node, -1
        else:
            return node, 0  # tie

    # Simulation: Play a random game from the generated child node using the default policy.
    # eps: During rollouts, the default policy may have a probability (ε) of choosing a random move rather than the best (so far) move.
    def rollout(self, node, eps=1, ANN=None):
        state = copy.copy(node.state)  # make a copy of the current state
        while not state.is_game_over():
            move = self.default_policy(state)
            state.do_move(move)
        return state.get_winner()

    # Backpropogation: updating relevant data (at all nodes and edges) on the path from the final state to the tree root node
    def backup(self, node, reward, child=None):
        node.visit_count += 1
        if child:
            child.q = child.compute_q(reward)
        if node.parent:
            self.backup(node.parent, reward, node)

    # må skille på om det er player 1 eller player 2?
    # Should use more rollouts in the beginning and then the critic more towards the end
    def default_policy(self, state, ANN=0, eps=1, stoch=True):
        moves = state.get_legal_moves()
        if random.random() < eps:
            return random.choice(moves)  # choose random action
        else:
            # ToDo: Må bruke et neural network her
            # ANN.get_action_probabilities(state) ?
            return random.choice(moves)

    # Tree policy: Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a).
    def tree_policy(self, node):
        children_stack = {}
        c = self.c if node.state.player else -self.c
        for child in node.children:
            children_stack[child] = child.q + node.compute_u(child, c)
        # player 1 choose branches with high Q values, player 2 choose those with low Q values.
        return max(children_stack, key=children_stack.get) if node.state.player else min(children_stack, key=children_stack.get)


if __name__ == '__main__':
    state = NimState(None, 2, 12, True)
    root = Node(state)
    mtcs = MCTS(NimState, root)  # input: game, root_node

    # M is number of MCTS simulations
    # For NIM and Ledge an M value of 500 is often sufficient
    # kan også bruke en time-limit:)) på ett-to sekunder for hvert kall på MCTS
    M = 500

    # tree search to get leaf node
    leaf_node = mtcs.get_leaf_node()
    # while not leaf_node.state.is_game_over():
    player1_win = 0
    player2_win = 0
    progress = []
    for i in range(M):
        print(f'The leaf node {leaf_node}')
        # expanding leaf_node
        mtcs.expand_node(leaf_node)
        # do rollout evaluation
        rollout_node, reward = mtcs.get_leaf_evalutation(leaf_node)
        # backpropogate exitevaluation
        mtcs.backup(rollout_node, reward)
        # Starting a new MTCS simulation
        leaf_node = mtcs.get_leaf_node()

        if leaf_node.state.is_game_over():
            winner = leaf_node.state.get_winner()
            if winner:
                player1_win += 1
                progress.append(1)
            else:
                player2_win += 1
                progress.append(2)

    # df = pd.DataFrame([player1_win, player2_win], [1, 2])
    # df.plot.bar()
    #df2 = pd.DataFrame(progress)
    # df2.plot()
    # plt.show()
    mtcs.root.visualize_tree()
