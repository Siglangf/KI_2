from ANET import ANET
from Simworld import Hex
import random

class TOPP:

    def __init__(self, board_size, game, num_games, episodes, num_agents):
        self.board_size = board_size
        self.anet = ANET(board_size)
        self.game = game
        self.agents = self.load_agents(episodes, num_agents)
        self.num_games = num_games

    
    def load_agents(self, episodes, num_agents):
        """
        Load all pre-trained actor neural networks into a list of agents
        :return: list[ANET]
        """
        players = []
        for num_games_trained in range(0,episodes+1,int(episodes/(num_agents-1))):
            players.append(self.anet.load_anet(str(self.board_size), str(num_games_trained)))
        return players

    def single_game(self, actors):
        """
        Plays one single game
        :param: actors = {1: p1, 2: p2}
        :return: int - winner
        """
        # Choose random player to start
        player = random.randint(1, 2)
        # init new game
        single_game = self.game(self.board_size, player)
        action_space = single_game.get_action_space()
        while not single_game.is_final():
            actions = single_game.legal_actions()
            current_state =  single_game.get_state()
            _, action_index = actors[player].get_move(current_state)
            action = action_space[action_index]
            
            single_game.step(action)
            player = single_game.player
    
        return single_game.get_winner()


    def run_tournament(self):
        """
        Performs a round robin tournament
        """
        wins = [0 for i in range(len(self.agents))]
        for i in range(len(self.agents)-1):
            for j in range(i+1, len(self.agents)):
                print(f'i: {i} og j: {j}')
                p1 = self.agents[i]
                p2 = self.agents[j]
                if i != j :
                    actors = {1: p1, 2: p2}
                    score = [0,0]
                    for _ in range(self.num_games):
                        winner = self.single_game(actors)
                        if winner==1:
                            wins[i]+=1
                        if winner==2:
                            wins[j]+=1
        return wins
                
                
                

    
if __name__ == '__main__':
    topp = TOPP(4, Hex, 10, 200, 5)
    topp.run_tournament()