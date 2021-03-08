# -----------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The Node class is the basic component of a Tree used for the Monte Carlo Tree Search.
# Object that keeps track of the non-static information about a game/episode e.g. number of boxes left
#
# -------------------------------------------------------IMPORTS-------------------------------------------------------
import math
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
# --------------------------------------------------------LOGIC--------------------------------------------------------


class Node:
    def __init__(self, state):
        self.parent = None
        # Liste med noder, kan children være en dictonary med key: child, value: [Q(s,a) + u(s,a)]?
        self.children = []
        self.state = state  # state object, e.g. for NIM number of pices left

        # Values that get updated through backward propagation of the MCTS
        self.q = 0
        self.visit_count = 0  # N(s) = number of visits to node
        self.total = 0

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    # Ultimate goal of MCTS: produce realistic Q(s,a) values
    # Q(s,a) = value (expected final result) of doing action a from state s.
    def compute_q(self, reward):
        self.total += reward
        #return (reward - self.q) / self.visit_count
        return self.total / self.visit_count

    # u(s,a)
    def compute_u(self, child, c):
        return c * math.sqrt(math.log(self.visit_count) / (1 + child.visit_count))

    # Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a)
    # Tree policy, whose behavior changes as Q(s,a) and u(s,a) values change
    # def get_sa_value(self, reward, child):
        # return self.compute_qsa(reward) + self.compute_usa(child)

    def __repr__(self):
        return f'Player: {1 if self.state.player else 2}, Pices left on board: {self.state.board}, Visits: {self.visit_count}, Value: {self.q}'

    def visualize_tree(self):
        G = nx.Graph()
        id_stack = [0]
        G.add_node(0)
        self.id = 0
        self.add_children_to_graph(self, G, id_stack)
        pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx(G, pos)
        plt.show()

    def add_children_to_graph(self, node, G, id_stack):
        if node.children:
            last_id = id_stack[-1]
            id_num = id_stack.pop(0)
            for i, child in enumerate(node.children):
                child_num = last_id+i+1
                G.add_node(child_num)
                G.add_edge(id_num, child_num)
                id_stack.append(id_num+child_num)
            for child in node.children:
                self.add_children_to_graph(child, G, id_stack)
