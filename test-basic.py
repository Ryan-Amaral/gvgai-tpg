from __future__ import division
import gym
import gym.spaces
import gym_gvgai

import time
import random
import pickle

import multiprocessing as mp
from multiprocessing.managers import BaseManager

from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent

# transforms the state into what the tpg agent can use.
def getState(state):
    state2 = []
    for x in state:
        for y in x:
            state2.append(y[0]/8 + y[1]*4 + y[2]*128)
            
    return state2



tStart = time.time()

allGames = ['gvgai-testgame1-lvl0-v0','gvgai-testgame1-lvl1-v0',
            'gvgai-testgame2-lvl0-v0','gvgai-testgame2-lvl0-v0',
            'gvgai-testgame3-lvl0-v0','gvgai-testgame3-lvl1-v0']
allGames = ['Assault-v0']

envs = {}
for game in allGames:
    envs[game] = gym.make(game)
        
gameQueue = list(allGames)
random.shuffle(gameQueue)
    
trainer = TpgTrainer(actions=range(6), teamPopSizeInit=360)
    
curScores = [] # hold scores in a generation
summaryScores = [] # record score summaries for each gen (min, max, avg)


for gen in range(100): # generation loop
    curScores = [] # new list per gen

    # get right env in envQueue
    game = gameQueue.pop() # take out last game
    print('playing on', game)
    env = envs[game]
    # re-get games list
    if len(gameQueue) == 0:
        gameQueue = list(allGames)
        random.shuffle(gameQueue)
    
    while True: # loop to go through agents
        teamNum = trainer.remainingAgents()
        agent = trainer.getNextAgent()
        if agent is None:
            break # no more agents, so proceed to next gen
        
        # check if agent already has score
        if agent.taskDone():
            score = agent.getOutcome()
        else:
            state = env.reset() # get initial state and prep environment
            score = 0
            valActs = range(env.action_space.n)
            for i in range(1000):

                act = agent.act(getState(state),valActs=valActs) # get action from agent

                # feedback from env
                state, reward, isDone, debug = env.step(act)
                score += reward # accumulate reward in score
                if isDone:
                    break # end early if losing state

            agent.reward(score) # must reward agent (if didn't already score)
            
        print('Agent #' + str(agent.getAgentNum()) + ' finished with score ' + str(score))
        curScores.append(score) # store score

    trainer.evolve()

    # save model after every gen
    with open('gvgai-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)
            
    # at end of generation, make summary of scores
    summaryScores.append((min(curScores), max(curScores),
                    sum(curScores)/len(curScores))) # min, max, avg
    
    print(chr(27) + "[2J")
    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('Results so far: ' + str(summaryScores))
    
#clear_output(wait=True)
print('Time Taken (Seconds): ' + str(time.time() - tStart))
print('Results:\nMin, Max, Avg')
for result in summaryScores:
    print(result[0],result[1],result[2])
