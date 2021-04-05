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
    counter = 0
    def __init__(self, state):
        self.parent = None # parent kan gi oss litt problemer siden vi hele tiden har en ny rot...
        self.children = [] # A lookup of legal child positions
        self.state = state  # The board state as this node, ex. for NIM number of pices left on the board.
        # self.to_play = to_play # The player whose turn it is. (1 or -1)

        # Values that get updated through backward propagation of the MCTS
        self.visit_count = 0  # N(s) = number of times this state was visited during MCTS. "Good" are visited more then "bad" states.
        self.value_sum = 0 # The total value of this state from all visits
        
        self.node_id = Node.counter 
        Node.counter +=  1
    
    def set_parent(self, node):
        self.parent = node

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)
    
    def expanded(self):
        return len(self.children) > 0

    # Ultimate goal of MCTS: produce realistic Q(s,a) values
    # Q(s,a) = value (expected final result) of doing action a from state s.
    def compute_q(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # u(s,a)
    def compute_u(self, c):
        if self.parent.visit_count == 0: #må fikses
            return float('inf')
        return c * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count + 1)) # math.log is the natural logarithm 

    def __repr__(self):
        #return f'ID: {self.node_id} Player: {1 if self.state.player else 2}, Pices left: {self.state.board}, Visits: {self.visit_count}'
        return f'ID: {self.node_id}, to_play: {self.state.player}, Pices left: {self.state.board}, Visits: {self.visit_count}, Value sum: {self.value_sum}'

    def visualize_tree(self):
        G = nx.Graph()
        global id_counter
        id_counter = 0
        G.add_node(0)
        self.id = 0
        self.add_children_to_graph(self, G)
        pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx(G, pos)
        plt.show()

    def add_children_to_graph(self, node, G):
        global id_counter
        if node.children:
            for i, child in enumerate(node.children):
                id_counter += 1
                child.id = id_counter
                G.add_node(child.id)
                G.add_edge(node.id, child.id)
            for child in node.children:
                self.add_children_to_graph(child, G)
