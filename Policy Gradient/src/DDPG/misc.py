from collections import deque, namedtuple
from matplotlib import pyplot as plt
import numpy as np
import random
import gym
import pdb
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
  env = gym.make("BipedalWalker-v3")
  env = gym.make(env_name)

  state_space = env.observation_space.shape[0]
  action_space = env.action_space.shape[0]

  print("State space: {}".format(state_space))
  print("Action space: {}".format(action_space))

  max_e = 300
  max_t = 700
  buffer_size = 100000
  batch_size = 32
  learn_every = 1

  gamma = 0.99
  tau = 1e-2

  score_log = []
  average_score_log = []
  score_window = deque(maxlen=100)

  # Create Agent
  agent = DDPG(state_space, action_space)

  for episode in range(max_e):
    state = env.reset()
    score = 0
    for t in range(max_t):
      action = DDPG.act(state, add_noise=False)
      next_state, reward, done, _ = env.step(action)
      mem.push(state, action, reward, next_state, done)
      score += reward


      
      if len(mem) > batch_size and t % learn_every == 0:
        learn()

      if done:
        break
      
      state = next_state
  
    score_log.append(score)
    score_window.append(score)
    average_score_log.append(np.mean(score_window))
    
    print("\rEpsiode: {:.1f}\tWindow Score: {:.4f}\tScore: {:.4f}".format(episode, np.mean(score_window), score), end="")    
    if (episode % 25 == 0):
      print("\rEpsiode: {:.1f}\tWindow Score: {:.4f}\tScore: {:.4f}".format(episode, np.mean(score_window), score))



