from ANET import ANET
from Simworld import Hex
import random


class TOPP:

    def __init__(self, series_name, board_size, game, num_games, episodes, num_agents):
        self.board_size = board_size
        self.game = game
        self.agents = self.load_agents(series_name, episodes, num_agents)
        self.num_games = num_games

    def load_agents(self, series_name, episodes, num_agents):
        """
        Load all pre-trained actor neural networks into a list of agents
        :return: list[ANET]
        """
        players = []
        for num_games_trained in range(0, episodes+1, int(episodes/(num_agents-1))):
            anet = ANET(self.board_size)
            anet.load_anet(series_name,
                           self.board_size, num_games_trained)
            players.append(anet)
        print(players)
        return players

    def single_game(self, actors):
        """
        Plays one single game
        :param: actors = {1: p1, 2: p2}
        :return: int - winner
        """
        # Choose random player to start
        player = 1
        # init new game

        single_game = self.game(self.board_size, player)
        action_space = single_game.get_action_space()
        while not single_game.is_final():
            current_state = single_game.get_state()
            _, action_index = self.agents[actors[player]].get_move(
                current_state)
            action = action_space[action_index]

            single_game.step(action)
            player = single_game.player
        winner = single_game.get_winner()
        return actors[winner]

    def run_tournament(self):
        """
        Performs a round robin tournament
        """
        wins = [0 for i in range(len(self.agents))]
        for i in range(len(self.agents)-1):
            for j in range(i+1, len(self.agents)):
                if i != j:
                    for game in range(self.num_games):
                        if game % 2 == 0:
                            actors = {1: i, 2: j}
                        else:
                            actors = {1: j, 2: i}
                        winner = self.single_game(actors)
                        wins[winner] += 1
        return wins


if __name__ == '__main__':
    topp = TOPP("siggi", 4, Hex, 10, 200, 5)
    topp.run_tournament()
