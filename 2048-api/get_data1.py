import pickle

from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent

for num in range(10):
    data = []
    for i in range(1000):
        game = Game(4, score_to_win=2048, random=False)
        agent = ExpectiMaxAgent(game)
        data += agent.playoutput()

    save_file = '2048_data/training_data' + str(num) + '.pkl'    
    training_data = open(save_file, 'wb')
    pickle.dump(data, training_data)

training_data.close()
