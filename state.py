# ------------------------------------------------------EXPLANATION-----------------------------------------------------
# DESCRIPTION:
# The GameManager implements keeps control of the current state of the game and implements state logic
# a) produces initial game states
# b) generates child states from a parent state
# c) recognizes winning states

# INPUT:
# -------------------------------------------------------IMPORTS--------------------------------------------------------
import copy
# --------------------------------------------------------LOGIC---------------------------------------------------------

class State:
    def __init__(self, board: int, player: int):
        self.board = board
        self.player = player

    def get_legal_moves(self):
        pass

    def is_game_over(self):
        pass

    def do_move(self):
        pass

    def get_winner(self):
        if self.is_game_over():
            return self.player


# N: number of pices on the board
# K: maximum number of pices to remove in one turn
# -: minimum of pices to remove in one turn is always 1
class NimState(State):

    def __init__(self, state, K = None, board = None, player = None):
        if state is None:
            super().__init__(board, player)
            self.K = K
        else:
            super().__init__(state.board, state.player)
            self.K = state.K

    def get_legal_moves(self):
        return [x for x in range(1, min(self.board, self.K) + 1)]

    #game is over when there are no pices left, aka. N = 0
    def is_game_over(self):
        return self.board == 0

    # move: number of pices to remove
    def do_move(self, move):
        self.board = self.board - move
        if not self.is_game_over(): #if game is over it is not the next palyers turn to move
            self.player = not self.player

    def __repr__(self):
        return f'Number of pices left is {self.board} and player {1 if self.player else 2}\'s turn'

# player 1 is True, player 2 is false
if __name__ == '__main__':
    board = NimState(None, 2, 7, True)











