import torch
import numpy as np
import pdb

class Memory:
  def __init__(self, size, num_env, env, device, gamma=0.99):
    self.gamma = gamma
    self.size = size
    self.device = device
    self.num_env = num_env
    self.observation_space = env.observation_space.shape 
    self.action_space = env.action_space.shape 

    self.states = torch.zeros((size, num_env) + env.observation_space.shape).to(device)
    self.actions = torch.zeros((size, num_env) + env.action_space.shape).to(device)
    self.rewards = torch.zeros((size, num_env)).to(device)
    self.log_probs = torch.zeros((size, num_env)).to(device)
    self.dones = torch.zeros((size, num_env)).to(device)

    self.idx = 0
    self.discounted_returns = None
  
  def add(self, states, actions, rewards, log_probs, dones):
    if (self.idx > self.size - 1):
      raise Exception("Memory out of space") 

    self.states[self.idx] = states
    self.actions[self.idx] = actions
    self.rewards[self.idx] = torch.FloatTensor(rewards.reshape(-1)).to(self.device)
    self.log_probs[self.idx] = torch.FloatTensor(log_probs.reshape(-1)).to(self.device)
    self.dones[self.idx] = torch.FloatTensor(dones.reshape(-1)).to(self.device)

    self.idx += 1
  
  def calculate_discounted_returns(self, last_value, next_done):
    # Create empty discounted returns array
    self.discounted_returns = torch.zeros((self.size, self.num_env)).to(self.device)

    for t in reversed(range(self.size)):
      # If first loop
      if t == self.size - 1:
        nextnonterminal = 1.0 - torch.FloatTensor(next_done).reshape(-1).to(self.device)
        next_return = torch.FloatTensor(last_value).reshape(-1).to(self.device)
      else:
        nextnonterminal = 1.0 - self.dones[t+1]
        next_return = self.discounted_returns[t+1]
      self.discounted_returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return

  def sample(self, mini_batch_idx):
    if self.discounted_returns is None:
      raise Exception("Calculate returns before sampling")
      
    # flatten into one array
    discounted_returns = self.discounted_returns.reshape(-1)
    states = self.states.reshape((-1,)+self.observation_space.shape)
    actions = self.actions.reshape((-1,)+self.action_space.shape)
    log_probs = self.log_probs.reshape(-1)

    # return sample
    return states[mini_batch_idx], actions[mini_batch_idx], log_probs[mini_batch_idx], discounted_returns[mini_batch_idx]
  
  def isFull(self):
    return self.idx == self.size
      
  def reset(self):
    self.idx = 0

    self.states = torch.zeros((size, num_env) + env.observation_space.shape).to(device)
    self.actions = torch.zeros((size, num_env) + env.action_space.shape).to(device)
    self.rewards = torch.zeros((size, num_env)).to(device)
    self.log_probs = torch.zeros((size, num_env)).to(device)
    self.dones = torch.zeros((size, num_env)).to(device)
  
  def get_mini_batch_idxs(self, mini_batch_size):
    idxs = np.arange(self.size*self.num_envs)
    np.random.shuffle(idxs)

    return [ idxs[start:start+mini_batch_size] for start in np.arange(0, self.size*self.num_envs, mini_batch_size)]
