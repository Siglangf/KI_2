from ANET import ANET
from Simworld import Hex
import random
import itertools
import numpy as np
import torch


class TOPP:

    def __init__(self, series_name, board_size, game, num_games, episodes, num_agents, hidden_layers):
        self.hidden_layers = hidden_layers
        self.board_size = board_size
        self.game = game
        self.agents = self.load_agents(series_name, episodes, num_agents)
        self.num_games = num_games
        self.scoreboard = self.generate_scoreboard()

    def load_agents(self, series_name, episodes, num_agents):
        """
        Load all pre-trained actor neural networks into a list of agents
        :return: list[ANET]
        """
        players = []
        for num_games_trained in range(0, episodes+1, int(episodes/(num_agents-1))):
            anet = ANET(input_size=self.board_size,
                        hidden_layers=self.hidden_layers)
            anet.load_anet(series_name,
                           self.board_size, num_games_trained)
            anet.model.eval()
            players.append(anet)
        return players

    def single_game(self, actors, greedy):
        """
        Plays one single game
        :param: actors = {1: p1, 2: p2}
        :return: int - winner
        """
        single_game = self.game(self.board_size)
        action_space = single_game.get_action_space()
        player = 1
        while not single_game.is_final():
            current_state = single_game.get_state()
            D, action_index = self.agents[actors[player]].get_move(current_state)
            action = action_space[action_index]
            if not greedy:
                D /= np.sum(D)
                if np.isnan(D).any():
                    D, action_index = self.agents[actors[player]].get_random_move(current_state, D)
                else:
                    action_index = np.random.choice(len(D), 1, p=D)[0]
                    action = action_space[action_index]
            single_game.step(action)
            player = single_game.player
        winner = single_game.get_winner()
        return actors[winner]

    def run_tournament(self, greedy = False):
        """
        Performs a round robin tournament
        """
        wins = [0 for i in range(len(self.agents))]
        opponents = list(itertools.permutations(range(len(self.agents)), 2))
        for player1, player2 in opponents:
            for _ in range(self.num_games):
                actors = {1: player1, 2: player2}
                winner = self.single_game(actors, greedy)
                wins[winner] += 1
                self.update_scoreboard(player1,player2,winner)
        return wins
    
    def generate_scoreboard(self):
        """ generates self.scoreboard """
        scoreboard = {}
        for p1_level in range(len(self.agents)):
            for p2_level in range(len(self.agents)):
                scoreboard[str(p1_level) + "-" + str(p2_level)] = [0, 0]     # (wins, losses)
        return scoreboard
    
    def update_scoreboard(self, p1_level, p2_level, winner):
        """ update self.scoreboard
        :param p1_id - int indicating id to player 1
        :param p2_id - int indicating id to player 2
        :param sp_won - boolean indicating if starting player won the game
        """
        if p1_level == winner:
            self.scoreboard[str(p1_level) + "-" + str(p2_level)][0] += 1
        else:
            self.scoreboard[str(p1_level) + "-" + str(p2_level)][1] += 1
    
    def print_scoreboard_results(self):
        """ prints self.scoreboard """
        print("\n\n      MATCHES   -----------------    RESULTS    ")
        print("      (P1, P2)   ---------------- (Wins, losses)")
        sorted(self.scoreboard.items(), key=lambda kv: (kv[0], kv[0]))
        for game in self.scoreboard.items():
            result = "({}, {})".format(game[1][0], game[1][1])
            print("       " + game[0] + "    -------------------   " + result)
