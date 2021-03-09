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
from Simworld import Hex
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------CONSTANTS---------------------------------------------------------
WIN = 2
TIE = 1
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
        if node.state.is_final():
            return node
        actions = node.state.legal_actions()
        children_nodes = []
        for action in actions:
            # make a copy of the current state
            child_state = copy.deepcopy(node.state)
            child_state.step(action)  # update the current state
            child_node = Node(child_state)
            children_nodes.append(child_node)
        node.add_children(children_nodes)

    # Leaf evaluation: Return 1 if player win, 0 if tie, else return -1
    def get_leaf_evaluation(self, node):
        # Do a rollout from child node, aka. Play a random game from on of the generated child nodes.
        if node.children:
            node = random.choice(node.children)
        return node, self.rollout(node)

    # Simulation: Play a random game from the generated child node using the default policy.
    # eps: During rollouts, the default policy may have a probability (ε) of choosing a random move rather than the best (so far) move.
    def rollout(self, node, eps=1, ANN=None):
        state = copy.copy(node.state)  # make a copy of the current state
        while not state.is_final():
            action = self.default_policy(state)
            state.step(action)
        return state.collect_reward()

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
        actions = state.legal_actions()
        if random.random() < eps:
            return random.choice(actions)  # choose random action
        else:
            # ToDo: Må bruke et neural network her
            # ANN.get_action_probabilities(state) ?
            return random.choice(actions)

    # Tree policy: Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a).
    def tree_policy(self, node):
        children_stack = {}
        c = self.c if node.state.player == 1 else -self.c
        for child in node.children:
            children_stack[child] = child.q + node.compute_u(child, c)
        # player 1 choose branches with high Q values, player 2 choose those with low Q values.
        return max(children_stack, key=children_stack.get) if node.state.player == 1 else min(children_stack, key=children_stack.get)


if __name__ == '__main__':
    state = Hex(4)
    #state = NimState(None, K=2, board=12, player=1)
    root = Node(state)
    mtcs = MCTS(Hex, root)  # input: game, root_node

    # M is number of MCTS simulations
    # For NIM and Ledge an M value of 500 is often sufficient
    # kan også bruke en time-limit:)) på ett-to sekunder for hvert kall på MCTS
    M = 10000

    # tree search to get leaf node
    leaf_node = mtcs.get_leaf_node()
    # while not leaf_node.state.is_final():
    player1_win = 0
    player2_win = 0
    draw = 0
    progress = []
    for i in range(M):
        print(f'The leaf node {leaf_node}')
        # expanding leaf_node
        mtcs.expand_node(leaf_node)
        # do rollout evaluation
        rollout_node, reward = mtcs.get_leaf_evaluation(leaf_node)
        # backpropogate evaluation
        mtcs.backup(rollout_node, reward)
        # Starting a new MTCS simulation
        leaf_node = mtcs.get_leaf_node()

        if leaf_node.state.is_final():
            # 2 represents a win. 1 Represent a tie (0 represents not finished)
            outcome = leaf_node.state.game_state()
            player = leaf_node.state.player
            print(player, outcome)
            if outcome == WIN and player == 1:
                player1_win += 1
                progress.append(1)
            elif outcome == WIN and player == 2:
                player2_win += 1
                progress.append(-1)
            else:
                draw += 1
                progress.append(0)

    df = pd.DataFrame([player1_win, player2_win, draw],
                      ["Player 1", "Player 2", "Draw"])
    df.plot.bar()
    df2 = pd.DataFrame(progress)
    df2.plot()
    plt.show()
    # mtcs.root.visualize_tree()
