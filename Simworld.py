import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np
NEIGHBOAR_MAPPING = {(0, 1): 'UP', (0, -1): 'DOWN', (-1, 0): 'LEFT',
                     (1, 0): 'RIGHT', (-1, 1): 'LEFT_DIAG', (1, -1): 'RIGHT_DIAG'}
COLOR_MAP = {0: "white", 1: "red", 2: "black"}


class Action:
    def __init__(self, action_cells):
        ''' Actioncells: the cell which is to be modified by the player'''
        self.action_cells = action_cells
        self.action_tuple = tuple(action_cells.position)


class Hex:
    def __init__(self, size, start_player=1):
        self.size = size
        # Player one represented as 1 and player two represented as 2
        self.player = 1
        self.board = Board(size)
        cells = self.board.cells
        # Defining owned sides of the board for each player
        self.sides = {1: (
            cells[0, :], cells[self.size-1, :]), 2: (cells[:, 0], cells[:, self.size-1])}

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

    def is_final(self, game_state=None):
        # If no game state is provided, use search function
        if game_state == None:
            game_state = self.game_state()
        return game_state > 0

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

    def get_winner(self):
        if self.is_final():
            return self.player
        print("Game is not finished")

    def collect_reward(self, game_state=None):
        if game_state == None:
            game_state = self.game_state()
        # If someone has won, and it's player one
        if game_state == 2 and self.player == 1:
            return 1
        # If someone has won, and it's player two
        if game_state == 2 and self.player == 2:
            return -1
        else:
            return 0

    def legal_actions(self):
        action_space = [Action(cell) for cell in self.board.get_empty_cells()]
        action_space = tuple([action.action_tuple for action in action_space])
        return action_space

    def get_action_space(self):
        empty_game = Hex(self.size, self.player)
        actions = empty_game.legal_actions()
        return {i: actions[i] for i in range(len(actions))}

    def step(self, action):
        action_cell = self.board.cell_from_position(action)
        action_cell.state = self.player
        state = self.game_state()
        is_final = self.is_final(state)
        reward = self.collect_reward(state)
        if not is_final:
            self.player = 1 if self.player == 2 else 2
        return self.board.to_tuple(), reward, is_final

    def action_to_string(self, action_cells):
        return None

    def get_state(self):
        board_flattened = np.hstack(self.board.to_tuple())
        state = np.insert(board_flattened, 0, self.player)
        return state

    def reset(self):
        self.__init__(self.size)

    def __repr__(self):
        return self.board.get_empty_cells()


class Cell:
    def __init__(self, state, position):
        self.state = state
        self.position = position

    def _repr__(self):
        return str(self.position)

    def __str__(self):
        return str(self.position)


class Board:
    def __init__(self, size):
        self.size = size
        # Initializing cell objects
        self.cells = np.array([[Cell(state=0, position=[j, i])
                                for i in range(self.size)] for j in range(self.size)])
        self.connect_adjacent()

    def get_empty_cells(self):
        return [cell for cell in np.hstack(self.cells) if cell.state == 0]

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

    def __repr__(self):
        return str(len(self.get_empty_cells()))


def visualize_state(environment, show_labels=False):
    Board = environment.board
    cells = np.hstack(Board.cells)
    G = nx.Graph()
    G.add_nodes_from(cells)
    for cell in cells:
        for pos, neighboar in cell.neighboars.items():
            G.add_edge(cell, neighboar)

    positions = {cell: [-10*cell.position[0] + 10*cell.position[1], -
                        20*cell.position[0]-20*cell.position[1]] for cell in cells}
    colors = []
    plt.ioff()
    for node in G:
        colors.append(COLOR_MAP[node.state])
    fig = plt.figure()
    nx.draw(G, pos=positions, ax=fig.add_subplot(),
            node_color=colors, with_labels=show_labels)

    fig = plt.figure(1)
    fig.axes[0].annotate('Player 2', xy=(0.26, 0.76),  xycoords='axes fraction',
                         xytext=(0.1, 0.99), textcoords='axes fraction',
                         arrowprops=dict(facecolor='black', shrink=0.1),
                         horizontalalignment='right', verticalalignment='top',
                         )
    fig.axes[0].annotate('Player 1', xy=(0.73, 0.76),  xycoords='axes fraction',
                         xytext=(0.99, 0.99), textcoords='axes fraction',
                         arrowprops=dict(facecolor='red', shrink=0.1),
                         horizontalalignment='right', verticalalignment='top',
                         )
    plt.close()
    if environment.is_final():
        if environment.player == 1:
            winner = "Blue player"
        elif environment.player == 2:
            winner = "Red player"
        else:
            winner = "No player"
        print(f"{winner} won the game")

    return fig
