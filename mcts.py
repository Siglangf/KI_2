# ------------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The MCTS class contains all the logic in order to perform Monte Carlo Tree Search

# INPUT:
# - root_node: root_node for the Monte Carlo Search Tree
# -------------------------------------------------------IMPORTS--------------------------------------------------------
import random
import numpy as np
from graph import Node,Tree
from state import State, NimState
# --------------------------------------------------------LOGIC---------------------------------------------------------

class MCTS:
    def __init__(self, state):
        self.state = state

    # Tree search: traversing the tree from the root node to a leaf node using the tree policy.
    # We start at the root and find a path to a frontier node by iteratively selecting the best child.
    def get_leaf_node(self, tree):
        node = tree.root
        while not node.state.is_game_over() and node.children:
            node = self.tree_policy(node)
        return node

    # Node expansion: Expand the selected node, that is, add one or more children to the node (usually only one).
    def expand_node(self, node):
        moves = node.state.get_legal_moves()
        children_nodes = []
        for move in moves:
            child_state = self.state(node.state) #make a copy of the current state
            child_state.do_move(move) #update the current state
            child_node = Node(child_state)
            children_nodes.append(child_node)
        node.add_children(children_nodes)
        # Do a rollout from child node, aka. Play a random game from on of the generated child nodes.
        # ToDo: Hvordan velge node til rollout ?
        child_node = random.choice(node.children)
        eval = self.get_leaf_evalutation(child_node)
        # Backup: Update the evaluation of all nodes on the path from the root to the generated child node based on the playout.
        self.backup(child_node, eval)
        return child_node

    # Leaf evaluation: Return 1 if player win, 0 if tie, else return -1
    def get_leaf_evalutation(self, node):
        winner = self.rollout(node)
        if node.state.player == winner: #player win
            return 1
        elif not node.state.player == winner: #opponent win
            return -1
        else:
            return 0 #tie

    # Simulation: Play a random game from the generated child node using the default policy.
    # eps: During rollouts, the default policy may have a probability (ε) of choosing a random move rather than the best (so far) move.
    def rollout(self, node, eps = 1, ANN = None):
        state = self.state(node.state) #make a copy of the current state
        while not state.is_game_over():
            move = self.default_policy(state)
            state.do_move(move)
        return state.get_winner()

    #Backpropogation: updating relevant data (at all nodes and edges) on the path from the final state to the tree root node
    def backup(self, node, eval, child = None):
        node.visit_count += 1
        # ToDo: Skal bare ta pluss når den vinner? ikke minus når den taper?
        if eval == 1:
            node.win_count += eval
        if child:
            node.update_sa_value(child)
        if node.parent:
            self.backup(node.parent, eval, node)

    # må skille på om det er player 1 eller player 2?
    # Should use more rollouts in the beginning and then the critic more towards the end
    def default_policy(self, state, NN=0, eps=1, stoch=True):
        moves = state.get_legal_moves()
        if random.random() < eps:
            return random.choice(moves)  # choose random action
        else:
            #ToDo: Må bruke et neural network her
            return random.choice(moves)

    # Tree policy: Choose branch with the highest combination of Q(s,a) + exploration bonus, u(s,a).
    def tree_policy(self, node):
        children_stack = {}
        for child in node.children:
            sa_value = child.value
            children_stack[child] = sa_value
        # player 1 choose branches with high Q values, player 2 choose those with low Q values.
        return max(children_stack, key=children_stack.get) if node.state.player else min(children_stack, key=children_stack.get)


if __name__ == '__main__':
    state = NimState(None, 3, 7, True)
    root_node = Node(state)
    mtcs = MCTS(NimState)
    tree = Tree(root_node)

    leaf_node = mtcs.get_leaf_node(tree)
    while not leaf_node.state.is_game_over():
        print(f'The node we are expanding is {leaf_node}')
        mtcs.expand_node(leaf_node)
        leaf_node = mtcs.get_leaf_node(tree)

    winner = leaf_node.state.get_winner()
    print(f'Game over, the winner is player {1 if winner else 2}')
