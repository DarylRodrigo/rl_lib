import copy
import gym
import torch
import numpy as np
from collections import deque
from PPO import PPO
from Config import Config


def train(config):
  env = copy.deepcopy(config.env)
  steps = 0
  scores_deque = deque(maxlen=100)
  scores = []
  max_score = -np.Inf

  agent = PPO(config)

  for i_episode in range(1, config.n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(config.max_t):
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

      if steps >= config.update_every:
        agent.learn(config.num_learn)
        agent.mem.clear()
        steps = 0

      if done:
        break 
      
    if i_episode % 10 == 0:
      print("\rEpisode {}	Average Score: {:.2f}	Score: {:.2f}".format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
      print("\rEpisode {}	Average Score: {:.2f}".format(i_episode, np.mean(scores_deque)))   
    
    if np.mean(scores_deque) > config.win_condition:
      print("\rEnvironment Solved!")
      break

  return scores
