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
        self.player = player #to_play

    def legal_actions(self):
        pass

    def is_final(self):
        pass

    def step(self):
        pass

    def game_state(self):
        pass


# N: number of pices on the board
# K: maximum number of pices to remove in one turn
# -: minimum of pices to remove in one turn is always 1
class NimState(State):

    def __init__(self, state, K=None, board=None, player=None):
        if state is None:
            super().__init__(board, player)
            self.K = K
        else:
            super().__init__(state.board, state.player)
            self.K = state.K

    def legal_actions(self):
        return [x for x in range(1, min(self.board, self.K) + 1)]

    # game is over when there are no pices left, aka. N = 0
    def is_final(self):
        return self.board == 0

    # return 0 if not ended, 1 if player 1 wins, -1 if player 1 lost
    def collect_reward(self): 
        if self.get_winner() == 1:
            return 1
        elif self.get_winner() == 2:
            return -1
        else:
            return 0

    def game_state(self):
        return 2

    # move: number of pices to remove
    def step(self, move):
        self.board = self.board - move
        self.player = 1 if self.player == 2 else 2

    def get_winner(self):
        if not self.is_final():
            return None
        else:
            return 1 if self.player == 2 else 2
            

    def __repr__(self):
        return f'Number of pices left is {self.board} and player {1 if self.player else 2}\'s turn'

