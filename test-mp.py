from __future__ import division
import gym
import gym.spaces
import gym_gvgai

import time
import random
import pickle
import psutil
import os

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

# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(19)

# run agent in function to work with multiprocessing
def runAgent(agentgmsq):
    agent = agentgmsq[0] # get agent
    game = agentgmsq[1] # get game name to create env
    sq = agentgmsq[2] # get score queue
    
    # check if agent already has score
    if agent.taskDone():
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        sq.put((agent.getUid(), agent.getOutcomes()))
        return
    
    #print('creating env')
    env = gym.make(game)
    #print('got env')
    state = env.reset() # get initial state and prep environment
    #print('reset env')
    
    valActs = range(env.action_space.n)
    score = 0
    for i in range(1000): # run episode
        act = agent.act(getState(state), valActs=valActs) # get action from agent

        # feedback from env
        state, reward, isDone, debug = env.step(act)
        score += reward # accumulate reward in score
        if isDone:
            break # end early if losing state

    env.close()
        
    agent.reward(score) # must reward agent
    
    print('Agent #' + str(agent.getAgentNum()) + ' finished with score ' + str(score))
    sq.put((agent.getUid(), agent.getOutcomes())) # get outcomes with id



tStart = time.time()
processes = 2 # how many to run concurrently
m = mp.Manager()

allGames = ['gvgai-testgame1-lvl0-v0','gvgai-testgame1-lvl1-v0',
            'gvgai-testgame2-lvl0-v0','gvgai-testgame2-lvl0-v0',
            'gvgai-testgame3-lvl0-v0','gvgai-testgame3-lvl1-v0']
allGames = ['Assault-v0']
        
gameQueue = list(allGames)
random.shuffle(gameQueue)
    
trainer = TpgTrainer(actions=range(6), teamPopSizeInit=360)

pool = mp.Pool(processes=processes, initializer=limit_cpu)
    
summaryScores = [] # record score summaries for each gen (min, max, avg)
    
for gen in range(100): # generation loop
    scoreQueue = m.Queue() # hold agents when finish, to actually apply score
    
    # get right env in envQueue
    game = gameQueue.pop() # take out last game
    print('playing on', game)
    # re-get games list
    if len(gameQueue) == 0:
        gameQueue = list(allGames)
        random.shuffle(gameQueue)
    
    # tasks = [str(envs[game][0].env)]
    
    # run generation
    pool.map(runAgent, 
                 [(agent, game, scoreQueue)#(agent, envQueue, scoreQueue)
                  for agent in trainer.getAllAgents(skipTasks=[])])
    
    scores = [] # convert scores into list
    while not scoreQueue.empty():
        scores.append(scoreQueue.get())

    # save model before every evolve in case issue
    with open('gvgai-model-1be.pkl','wb') as f:
        pickle.dump(trainer,f)
    
    # apply scores
    trainer.applyScores(scores)
    trainer.evolve() # go into next gen
    
    # save model after every gen
    with open('gvgai-model-1ae.pkl','wb') as f:
        pickle.dump(trainer,f)

    # at end of generation, make summary of scores
    summaryScores.append((trainer.scoreStats['min'], 
                    trainer.scoreStats['max'],
                    trainer.scoreStats['average'])) # min, max, avg
    print(chr(27) + "[2J")
    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('Results so far: ' + str(summaryScores))
  
print(chr(27) + "[2J")  
print('Time Taken (Seconds): ' + str(time.time() - tStart))
print('Results: ' + str(summaryScores))

# 451.15136647224426 seconds for pop 50, frames 50

