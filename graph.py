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

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    # Ultimate goal of MCTS: produce realistic Q(s,a) values
    # Q(s,a) = value (expected final result) of doing action a from state s.
    def compute_q(self, reward):
        return  (reward - self.q) / self.visit_count

    # u(s,a)
    def compute_u(self, child, c):
        return c * math.sqrt( math.log(self.visit_count) / (1 + child.visit_count))

    # Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a)
    # Tree policy, whose behavior changes as Q(s,a) and u(s,a) values change
    # def get_sa_value(self, reward, child):
        # return self.compute_qsa(reward) + self.compute_usa(child)

    def __repr__(self):
        return f'Player: {1 if self.state.player else 2}, Pices left on board: {self.state.board}, Visits: {self.visit_count}, Value: {self.q}'


    def visualize_tree(self):
        G = nx.Graph()
        global counter
        counter = 0
        G.add_node(0)
        self.add_children_to_graph(self,G, counter)
        pos = graphviz_layout(G, prog="dot")
        nx.draw_networkx(G)
        plt.show()
    
    def add_children_to_graph(self,node,G,counter):
        for child_num in range(1,len(node.children)+1):
            print(counter,counter+child_num)
            G.add_node(counter+child_num)
            G.add_edge(counter,counter+child_num)
        for child in range(len(node.children)):
            counter+=1
            self.add_children_to_graph(node.children[child],G,counter)
    

 
        