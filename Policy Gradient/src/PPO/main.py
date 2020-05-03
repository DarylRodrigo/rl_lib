import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PPO import PPO

env = gym.make('CartPole-v1')
env.seed(10)

state_size = env.observation_space.shape[0]
action_size =env.action_space.n

# PPO Settings
update_every = 2000
num_learn = 4
win_condition = 230

def train(n_episodes=2000, max_t=700):
  steps = 0
  scores_deque = deque(maxlen=100)
  scores = []
  max_score = -np.Inf

  agent = PPO(state_size, action_size)

  for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(max_t):
      steps += 1

      action, log_prob = agent.act(torch.FloatTensor(state))
      next_state, reward, done, _ = env.step(action.item())

      agent.mem.add(torch.FloatTensor(state), action, reward, log_prob, done)

      # Update 
      state = next_state
      score += reward

      # Book Keeping
      scores_deque.append(score)
      scores.append(score)

      if steps >= update_every:
        agent.learn(num_learn)
        agent.mem.clear()
        steps = 0

      if done:
        break 
      
    if i_episode % 10 == 0:
      print("\rEpisode {}	Average Score: {:.2f}	Score: {:.2f}".format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
      print("\rEpisode {}	Average Score: {:.2f}".format(i_episode, np.mean(scores_deque)))   
    
    if np.mean(scores_deque) > win_condition:
      print("\rEnvironment Solved!")
      break


  return scores

scores = train()