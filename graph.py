# -----------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The Node class is the basic component of a Tree used for the Monte Carlo Tree Search.
# Object that keeps track of the non-static information about a game/episode e.g. number of boxes left
#
# -------------------------------------------------------IMPORTS-------------------------------------------------------
import math
# --------------------------------------------------------LOGIC--------------------------------------------------------


class Node:
    def __init__(self, state):
        self.win_count = 0
        self.parent = None
        self.children = [] # Liste med noder, kan children være en dictonary med key: child, value: [Q(s,a) + u(s,a)]?
        self.state = state #state object, e.g. for NIM number of pices left
        
        # Values that get updated through backward propagation of the MCTS
        self.value = 0 # tree policy value
        self.visit_count = 0 # N(s) = number of visits to node

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    # Ultimate goal of MCTS: produce realistic Q(s,a) values
    # Q(s,a) = value (expected final result) of doing action a from state s.
    def compute_qsa(self):
        return self.win_count / self.visit_count

    # u(s,a)
    def compute_usa(self, child):
        return math.sqrt(math.log(self.visit_count)) / (1 + child.visit_count)

    # Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a)
    # Tree policy, whose behavior changes as Q(s,a) and u(s,a) values change
    def get_sa_value(self,child):
        return self.compute_qsa() + self.compute_usa(child)

    def update_sa_value(self, child):
        child.value = self.get_sa_value(child)

    def __repr__(self):
        return f'Player: {1 if self.state.player else 2}, Pices left on board: {self.state.board}, Visits: {self.visit_count}, Value: {self.value}'




