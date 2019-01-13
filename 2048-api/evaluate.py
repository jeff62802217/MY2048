from game import Game
from displays import Display
import time
import MyAgent

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    game = Game(GAME_SIZE, SCORE_TO_WIN, random=False)

    '''====================
    Use your own agent here.'''
    TestAgent = MyAgent.MyAgent
    '''===================='''

    scores = []
    Time = [time.asctime( time.localtime(time.time()) )]
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)
    Time.append(time.asctime( time.localtime(time.time()) ))
    ave = sum(scores) / len(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    print(scores)
    for i in Time:
        print(i)
