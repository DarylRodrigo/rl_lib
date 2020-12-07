import copy
import gym
import torch
import numpy as np
from collections import deque
from PPO import PPOClassical, PPOPixel
from Config import Config
import pdb

def train(config):
  env = copy.deepcopy(config.env)
  steps = 0
  scores_deque = deque(maxlen=100)
  scores = []
  average_scores = []
  max_score = -np.Inf

  agent = PPOClassical(config)

  for i_episode in range(1, config.n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(config.max_t):
      steps += 1

      action, log_prob = agent.act(torch.FloatTensor(state))
      next_state, reward, done, _ = env.step(action.item())

      agent.add_to_mem(state, action, reward, log_prob, done)

      # Update 
      state = next_state
      score += reward

      if steps >= config.update_every:
        agent.learn(config.num_learn)
        agent.mem.clear()
        steps = 0

      if done:
        break 

    # Book Keeping
    scores_deque.append(score)
    scores.append(score)
    average_scores.append(np.mean(scores_deque))
    
      
    if i_episode % 10 == 0:
      print("\rEpisode {}	Average Score: {:.2f}	Score: {:.2f}".format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
      print("\rEpisode {}	Average Score: {:.2f}".format(i_episode, np.mean(scores_deque)))   
    
    if np.mean(scores_deque) > config.win_condition:
      print("\nEnvironment Solved!")
      break

  return scores, average_scores

def train_pixel(config):
  env = copy.deepcopy(config.env)
  steps = 0
  scores_deque = deque(maxlen=100)
  scores = []
  average_scores = []
  max_score = -np.Inf
  global_step = 0

  agent = PPOPixel(config)

  while global_step < 10000000:
    state = env.reset()
    score = 0
    for t in range(config.update_every):
      steps += 1
      global_step += 1

      action, log_prob, value, entr = agent.act(state)
      next_state, reward, done, info = env.step(action)
      agent.add_to_mem(state, action, reward, log_prob, done)

      # Update 
      state = next_state
      score += reward

      if steps >= config.update_every:
        value_loss, pg_loss, approx_kl = agent.learn(config.num_learn)
        agent.mem.clear()
        steps = 0
      

    # Book Keeping
    scores_deque.append(score)
    scores.append(score)
    average_scores.append(np.mean(scores_deque))

    config.tb_logger.add_scalar("charts/episode_reward", score, global_step)
    config.tb_logger.add_scalar("losses/value_loss", value_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

    print("Global Step: {}	Average Score: {:.2f}".format(global_step, np.mean(scores_deque)))  

  return scores, average_scores


