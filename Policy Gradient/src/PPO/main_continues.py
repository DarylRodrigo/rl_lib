
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PPO import PPO, PPOContinuous
import pdb

env = gym.make("Pendulum-v0")
# env = gym.make("MountainCarContinuous-v0")

env.seed(10)

state_size = env.observation_space.shape[0]
action_size =env.action_space.shape[0]

print(state_size)
print(action_size)

# PPO Settings
update_every = 2000
num_learn = 40
win_condition = 0

# Agent settings
hidden_size=128
epsilon=0.2
entropy_beta=0.01
gamma=0.99
lr=0.0003

agent = PPOContinuous(state_size, action_size, hidden_size=hidden_size, epsilon=epsilon, entropy_beta=entropy_beta, gamma=gamma, lr=lr)

def train(n_episodes=4000, max_t=1500):
  steps = 0
  scores_deque = deque(maxlen=100)
  scores = []
  average_scores = []
  max_score = -np.Inf

#   agent = PPO(state_size, action_size, hidden_size=hidden_size, epsilon=epsilon, entropy_beta=entropy_beta, gamma=gamma, lr=lr)

  for episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    
    for t in range(max_t):
      steps += 1

      actions_tensor, log_prob = agent.act(torch.FloatTensor(state))
      actions = actions_tensor.cpu().data.numpy().flatten()
      next_state, reward, done, _ = env.step(actions_tensor)

      agent.mem.add(torch.FloatTensor(state), actions, reward, log_prob, done)

      # Update 
      state = next_state
      score += reward.item()

      if steps >= update_every:
        agent.learn(num_learn)
        agent.mem.clear()
        steps = 0

      if done:
        break
    
    # Book Keeping
    scores_deque.append(score)
    scores.append(score)
    average_scores.append(np.mean(scores_deque))
      
    if episode % 10 == 0:
      print("\rEpisode {}	Average Score: {:.2f}	Score: {:.2f}".format(episode, np.mean(scores_deque), score), end="")
    if episode % 100 == 0:
      print("\rEpisode {}	Average Score: {:.2f}".format(episode, np.mean(scores_deque)))   
    
    if np.mean(scores_deque) > win_condition:
      print("\rEnvironment Solved in {} episodes!".format(episode))
      break


  return scores, average_scores

scores, average_scores = train()