from __future__ import division
import gym
import gym.spaces
import gym_gvgai

import time
import random
import pickle

import multiprocessing as mp

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
def runAgent(agenteqsq):
    agent = agenteqsq[0] # get agent
    eq = agenteqsq[1] # get environment queue
    sq = agenteqsq[2] # get score queue
    
    # check if agent already has score
    if agent.taskDone():
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        sq.put((agent.getUid(), agent.getOutcomes()))
        return
        
    print('envs in queue:',eq.qsize())
    env = eq.get() # get an environment
    state = env.reset() # get initial state and prep environment
    
    valActs = range(env.action_space.n)
    score = 0
    for i in range(50): # run episodes that last 200 frames
        act = agent.act(getState(state), valActs=valActs) # get action from agent
        
        # feedback from env
        state, reward, isDone, debug = env.step(act)
        score += reward # accumulate reward in score
        if isDone:
            break # end early if losing state
        
    agent.reward(score) # must reward agent
    
    print('Agent #' + str(agent.getAgentNum()) + ' finished with score ' + str(score))
    sq.put((agent.getUid(), agent.getOutcomes())) # get outcomes with id
    eq.put(env) # put environment back




tStart = time.time()
processes = 3 # how many to run concurrently (3 is best for my local desktop)
m = mp.Manager()

allGames = ['gvgai-testgame1-lvl0-v0','gvgai-testgame1-lvl1-v0',
            'gvgai-testgame2-lvl0-v0','gvgai-testgame2-lvl0-v0',
            'gvgai-testgame3-lvl0-v0','gvgai-testgame3-lvl1-v0']
allGames = ['Assault-v0']

envs = {}
for game in allGames:
    envs[game] = []
    for p in range(processes): # each process needs its own environment
        envs[game].append(gym.make(game))
        
gameQueue = list(allGames)
random.shuffle(gameQueue)
    
trainer = TpgTrainer(actions=range(6), teamPopSizeInit=50)

pool = mp.Pool(processes=processes)
    
summaryScores = [] # record score summaries for each gen (min, max, avg) 
    
for gen in range(100): # generation loop
    scoreQueue = m.Queue() # hold agents when finish, to actually apply score
    envQueue = m.Queue() # hold envs for current gen
    
    # get right env in envQueue
    game = gameQueue.pop() # take out last game
    print('playing on', game)
    for p in range(processes):
        envQueue.put(envs[game][p])
    # re-get games list
    if len(gameQueue) == 0:
        gameQueue = list(allGames)
        random.shuffle(gameQueue)
        
    # tasks = [str(envs[game][0].env)]
    
    # run generation
    # skipTasks=[] so we get all agents, even if already scored,
    # just to report the obtained score for all agents.
    pool.map(runAgent, 
                 [(agent, envQueue, scoreQueue) 
                  for agent in trainer.getAllAgents(skipTasks=[])])
    
    scores = [] # convert scores into list
    while not scoreQueue.empty():
        scores.append(scoreQueue.get())
    
    # apply scores
    trainer.applyScores(scores)
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

# 451.15136647224426 seconds for pop 50, frames 50

