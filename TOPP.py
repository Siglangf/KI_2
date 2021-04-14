from ANET import ANET
from Simworld import Hex, visualize_game_jupyter, visualize_state, show_visualization_jupyter
import random
import itertools
import numpy as np
import torch
import time


class TOPP:

    def __init__(self, series_name, board_size, game, num_games, episodes, num_agents, hidden_layers):
        self.hidden_layers = hidden_layers
        self.board_size = board_size
        self.game = game
        self.level_mapping = {}
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
            self.level_mapping[len(players)-1] = num_games_trained
        return players

    def single_game(self, actors, greedy, visualize=False):
        """
        Plays one single game
        :param: actors = {1: p1, 2: p2}
        :return: int - winner
        """
        single_game = self.game(self.board_size)
        action_space = single_game.get_action_space()
        player = 1
        game_frames = []
        while not single_game.is_final():
            if visualize:
                frame = visualize_state(single_game)
                game_frames.append(frame)
            current_state = single_game.get_state()
            D, action_index = self.agents[actors[player]].get_move(
                current_state)
            action = action_space[action_index]
            if not greedy:
                D /= np.sum(D)
                if np.isnan(D).any():
                    D, action_index = self.agents[actors[player]].get_random_move(
                        current_state, D)
                else:
                    action_index = np.random.choice(len(D), 1, p=D)[0]
                    action = action_space[action_index]
            single_game.step(action)
            player = single_game.player
        if visualize:
            frame = visualize_state(single_game)
            game_frames.append(frame)
        winner = single_game.get_winner()
        return actors[winner], game_frames

    def run_tournament(self, greedy=False, visualize=False):
        """
        Performs a round robin tournament
        """
        wins = [0 for i in range(len(self.agents))]
        opponents = list(itertools.permutations(range(len(self.agents)), 2))
        for player1, player2 in opponents:
            for _ in range(self.num_games):
                actors = {1: player1, 2: player2}
                winner, game_frames = self.single_game(
                    actors, greedy, visualize)
                wins[winner] += 1
                self.update_scoreboard(player1, player2, winner)
                if visualize:
                    show_visualization_jupyter(game_frames[-1])
                    print(
                        f"PLAYER 1: level {self.level_mapping[player1]} VS PLAYER 2: level {self.level_mapping[player2]}. LEVEL {self.level_mapping[winner]} wins!!!!!!")
                    input("")

        return wins

    def generate_scoreboard(self):
        """ generates self.scoreboard """
        scoreboard = {}
        for p1_level in range(len(self.agents)):
            for p2_level in range(len(self.agents)):
                scoreboard[str(self.level_mapping[p1_level]) + "-" + str(self.level_mapping[p2_level])
                           ] = [0, 0]     # (wins, losses)
        return scoreboard

    def update_scoreboard(self, p1_level, p2_level, winner):
        """ update self.scoreboard
        :param p1_id - int indicating id to player 1
        :param p2_id - int indicating id to player 2
        :param sp_won - boolean indicating if starting player won the game
        """
        if p1_level == winner:
            self.scoreboard[str(self.level_mapping[p1_level]) +
                            "-" + str(self.level_mapping[p2_level])][0] += 1
        else:
            self.scoreboard[str(self.level_mapping[p1_level]) +
                            "-" + str(self.level_mapping[p2_level])][1] += 1

    def print_scoreboard_results(self):
        """ prints self.scoreboard """
        print("\n\n      MATCHES   -----------------    RESULTS    ")
        print("      (P1, P2)   ---------------- (Wins, losses)")
        sorted(self.scoreboard.items(), key=lambda kv: (kv[0], kv[0]))
        for game in self.scoreboard.items():
            result = "({}, {})".format(game[1][0], game[1][1])
            print("       " + game[0] + "    -------------------   " + result)

    def play_random(self,  agent, choice="argmax"):
        # For test resons
        starting_player_id = 1
        wins = 0
        for i in range(self.num_games):
            starting_player_id = 1 if starting_player_id == 2 else 2
            single_game = self.game(self.board_size, starting_player_id)
            action_space = single_game.get_action_space()
            while not single_game.is_final():
                if single_game.player == 1:
                    current_state = single_game.get_state()
                    if starting_player_id == 2:
                        current_state[0] = 2
                    D, action_index = self.agents[agent].get_move(
                        current_state)
                    action = action_space[action_index]
                if single_game.player == 2:
                    action_list = single_game.legal_actions()
                    action = random.sample(action_list, 1)[0]
                single_game.step(action)
            winner = single_game.get_winner()
            if winner == 1:
                wins += 1
        return wins/self.num_games
