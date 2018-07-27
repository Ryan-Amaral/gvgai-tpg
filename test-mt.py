from __future__ import division
import gym
import gym.spaces
import gym_gvgai

import time
import random
import pickle

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent

# transforms the state into what the tpg agent can use.
def getState(state):
    state2 = []
    for x in state:
        for y in x:
            state2.append(y[0]/8 + y[1]*4 + y[2]*128)
            
    return state2

# run agent in function to work with multiprocessing
def runAgent(agent, envQueue):

    lock.acquire()
    print('envs in queue:', len(envQueue))
    env = envQueue.pop() # get an environment
    lock.release()

    state = env.reset() # get initial state and prep environment
    
    valActs = range(env.action_space.n)
    score = 0
    for i in range(50): # run episodes that last 200 frames
        act = agent.act(getState(state), valActs=valActs) # get action from agent
        print(i)
        # feedback from env
        state, reward, isDone, debug = env.step(act)
        score += reward # accumulate reward in score
        if isDone:
            break # end early if losing state
        
    lock.acquire()
    agent.reward(score) # must reward agent
    lock.release()
    
    print('Agent #' + str(agent.getAgentNum()) + ' finished with score ' + str(score))
    
    lock.acquire()
    eq.put(env) # put environment back
    lock.release()




tStart = time.time()
workers = 10 # how many to run concurrently (3 is best for my local desktop)
lock = Lock()

allGames = ['gvgai-testgame1-lvl0-v0','gvgai-testgame1-lvl1-v0',
            'gvgai-testgame2-lvl0-v0','gvgai-testgame2-lvl0-v0',
            'gvgai-testgame3-lvl0-v0','gvgai-testgame3-lvl1-v0']
allGames = ['Assault-v0']

envs = {}
for game in allGames:
    envs[game] = []
    for w in range(workers): # each process needs its own environment
        envs[game].append(gym.make(game))
        
gameQueue = list(allGames)
random.shuffle(gameQueue)
    
trainer = TpgTrainer(actions=range(6), teamPopSizeInit=50)
    
summaryScores = [] # record score summaries for each gen (min, max, avg)
    
for gen in range(100): # generation loop
    envQueue = [] # hold envs for current gen
    
    # get right env in envQueue
    game = gameQueue.pop() # take out last game
    print('playing on', game)
    for w in range(workers):
        envQueue.append(envs[game][w])
    # re-get games list
    if len(gameQueue) == 0:
        gameQueue = list(allGames)
        random.shuffle(gameQueue)
        
    # tasks = [str(envs[game][0].env)]
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
	    {executor.submit(runAgent, agent, envQueue): agent for agent in trainer.getAllAgents()}
    
    trainer.evolve() # go into next gen
    
    # save model after every gen
    with open('gvgai-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)

    # at end of generation, make summary of scores
    summaryScores.append((trainer.scoreStats['min'], 
                    trainer.scoreStats['max'],
                    trainer.scoreStats['average'])) # min, max, avg

    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('Results so far: ' + str(summaryScores))
    
print('Time Taken (Seconds): ' + str(time.time() - tStart))
print('Results: ' + str(summaryScores))

# to many seconds for pop 50, frames 50

