import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym

import datetime
import time

from src.agents.DQN import Agent

def train(config, logger):
  experiment_start = time.time()
  env = config.env

  agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, config=config)

  total_scores_deque = deque(maxlen=100)
  total_scores = []


  for i_episode in range(1, config.n_episodes+1):
    states = env.reset()
    scores = 0
    
    start_time = time.time()
    
    for t in range(config.max_t):
      # actions = agent.act(states)
      # next_states, rewards, dones, _ = env.step(np.array(actions))
      # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
      #   agent.step(state, action, reward, next_state,done)

      actions = agent.act(states)
      next_states, rewards, dones, _ = env.step(np.array(actions))
      agent.step(states, actions, rewards, next_states,dones)

      
      
      states = next_states
      scores += rewards
      number_of_time_steps = t
      
      if np.any(dones):
        break 
    
    agent.anneal_eps()


    # Book Keeping
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    total_scores_deque.append(mean_score)
    total_scores.append(mean_score)
    total_average_score = np.mean(total_scores_deque)

    logger.log_scalar("score", mean_score, i_episode)
    
    duration = time.time() - start_time

    # print('\rEpisode {}\tTotal Average Score (in 100 window): {:.4f}\tMean: {:.4f}\tMin: {:.4f}\tMax: {:.4f}\tDuration: {:.4f}\t#TimeSteps: {:.4f}'.format(i_episode, total_average_score, mean_score, min_score, max_score, duration, number_of_time_steps), end="")
    print('\rEpi: {}\tAverage Score: {:.4f}\tMean: {:.4f}\tDuration: {:.4f}\t#t_s: {:.4f}'.format(i_episode, total_average_score, mean_score, duration, number_of_time_steps), end="")

    if i_episode % 100 == 0:
      print('\rEpi: {}\tAverage Score: {:.4f}\tMean: {:.4f}\tDuration: {:.4f}\t#t_s: {:.4f}'.format(i_episode, total_average_score, mean_score, duration, number_of_time_steps))
      torch.save(agent.qnetwork_local.state_dict(), "{}/checkpoint.pth".format(logger.log_file_path, date=datetime.datetime.now()))
    if config.win_condition is not None and total_average_score > config.win_condition: 
      print("\nEnvironment Solved in {:.4f} seconds !".format(time.time() - experiment_start))
      torch.save(agent.qnetwork_local.state_dict(), "{}/checkpoint.pth".format(logger.log_file_path, date=datetime.datetime.now()))
      return 
                
  return

def watch(config, log_file_path):
  # load the weights from file
  env = config.env
  agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, config=config)
  agent.qnetwork_local.load_state_dict(torch.load(log_file_path+'/checkpoint.pth'))

  for i in range(3):
    r=0
    state = env.reset()
    env.render(mode='rgb_array')
    for j in range(config.max_t):
      action = agent.act(state)
      env.render()
      state, reward, done, _ = env.step(action)

      if done:
        break 
            
  env.close()

def watch_untrained():
  env = config.env
  agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, config=config)
  state = env.reset()
  
  for j in range(1000):
    action = agent.act(state, network_only=True)
    state, reward, done, _ = env.step(action)
    if np.any(done):
      break 
        
  env.display()