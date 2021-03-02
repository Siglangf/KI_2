import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np
NEIGHBOAR_MAPPING = {(0, 1): 'UP', (0, -1): 'DOWN', (-1, 0): 'LEFT',
                     (1, 0): 'RIGHT', (-1, 1): 'LEFT_DIAG', (1, -1): 'RIGHT_DIAG'}
COLOR_MAP = {(0, 0): "white", (1, 0): "blue", (0, 1): "red"}


class Action:
    def __init__(self, action_cells):
        ''' Actioncells: the cell which is to be modified by the player'''
        self.action_cells = action_cells
        self.action_tuple = tuple(action_cells.position)


class Hex:
    def __init__(self, board_type, size, open_cells):
        self.board_type = board_type
        self.size = size
        self.open_cells = open_cells
        # Player 1 represented as (1,0), and player two represented as (0,1)
        self.player = (1, 0)
        self.board = Board(board_type, size, open_cells)
        cells = self.board.cells
        # Defining owned sides of the board for each player
        self.sides = {(1, 0): (
            cells[0, :], cells[self.size-1, :]), (0, 1): (cells[:, 0], cells[:, self.size-1])}

    def game_state(self):
        '''Returns 2 if current player has won, 1 if the games is over and it's draw, 0 if the games is not over'''
        # All the cells from one side
        search_cells = [cell for cell in self.sides[self.player]
                        [0] if cell.state == self.player]
        visited = set()
        # Depth first search from each of the cells on one side, trying to find a neighboar cell on the other
        for cell in search_cells:
            if self.search_connected(cell, visited):
                return 2
        if len(self.legal_actions()) == 0:
            return 1
        return 0

    def search_connected(self, cell, visited):
        '''Depth first search, finding consecutive lines across each side'''
        # Adding cell to visited
        visited.add(cell)
        # Returns true if the cell is on the opposite side
        if cell in self.sides[self.player][1]:
            return True
        # Get neighboars where pieces are from the same player, and not yet searched
        neighboars = [neig for neig in cell.neighboars.values(
        ) if neig.state == self.player and neig not in visited]
        if len(neighboars) == 0:
            return False

        # Continue search from all of the neighboars
        for neig in neighboars:
            if self.search_connected(neig, visited):
                return True
        return False

    def collect_reward(self, game_state):
        if game_state == 2:
            return 10
        else:
            return 0

    def legal_actions(self):
        action_space = [Action(cell) for cell in self.board.get_empty_cells()]
        action_space = tuple([action.action_tuple for action in action_space])
        return action_space

    def step(self, action):
        action_cell = self.board.cell_from_position(action)
        action_cell.state = self.player
        state = self.game_state()
        is_final = state > 0
        reward = self.collect_reward(state)
        self.player = (0 if self.player[0] else 1, 0 if self.player[1] else 1)
        return self.board.to_tuple(), reward, is_final

    def action_to_string(self, action_cells):
        return None

    def get_state(self):
        return self.board.to_tuple()

    def reset(self):
        self.__init__(self.board_type, self.size, self.open_cells)


class Cell:
    def __init__(self, state, position):
        self.state = state
        self.position = position

    def _repr__(self):
        state = "Pegged" if self.state else "Empty"
        return f"{state} cell at position ({self.position[0]},{self.position[1]})"

    def __str__(self):
        return str(self.position)


class Board:
    def __init__(self, board_type, size, open_cells):
        self.board_type = board_type
        self.size = size
        self.open_cells = open_cells

        # Initializing cell object
        if board_type == "triangle":
            self.cells = np.array([[Cell((0, 0), [j, i]) for i in range(
                size-j)] for j in range(self.size)], dtype=object)
        elif board_type == "diamond":

            self.cells = np.array(
                [[Cell((0, 0), [i, j]) for i in range(self.size)] for j in range(self.size)])
        else:
            raise ValueError("Board type must be 'triangle' or 'diamond'")

        self.connect_adjacent()

    def get_empty_cells(self):
        return [cell for cell in np.hstack(self.cells) if cell.state == (0, 0)]

    def connect_adjacent(self):
        '''Connect adjecent cells to each other in cell.neighboars'''
        cells = self.cells
        for i in range(len(cells)):
            for j in range(len(cells[i])):
                cells[i][j].neighboars = {}
                for delta, position in NEIGHBOAR_MAPPING.items():
                    adjx = i+delta[0]
                    adjy = j+delta[1]
                    # Avoiding negative index referances
                    if adjx < 0 or adjy < 0:
                        continue
                    try:
                        cells[i][j].neighboars[position] = cells[adjx][adjy]
                    except:
                        continue

    def cell_from_position(self, position):
        cells = np.hstack(self.cells)
        for cell in cells:
            if cell.position[0] == position[0] and cell.position[1] == position[1]:
                return cell

    def to_tuple(self):
        l = []
        for row in self.cells:
            l.append(tuple([cell.state for cell in row]))
        return tuple(l)


def visualize_state(environment):
    Board = environment.board
    cells = np.hstack(Board.cells)
    G = nx.Graph()
    G.add_nodes_from(cells)
    for cell in cells:
        for pos, neighboar in cell.neighboars.items():
            G.add_edge(cell, neighboar)

    positions = {cell: [cell.position[0], cell.position[1]] for cell in cells}
    colors = []
    plt.ioff()
    for node in G:
        colors.append(COLOR_MAP[node.state])
    fig = plt.figure()
    nx.draw(G, pos=positions, ax=fig.add_subplot(), node_color=colors)
    plt.close()
    return fig
