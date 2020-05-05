import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import matplotlib.pyplot as plt

import pdb

class ActorCriticContinuous(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, state_space, action_space, hidden_size):
    super(ActorCriticContinuous, self).__init__()

    self.action_space = action_space
    self.state_space = state_space
    
    # self.head = nn.Sequential(
    #   nn.Linear(state_space, hidden_size),
    #   nn.Tanh()
    # )

    # self.actor = nn.Sequential(
    #   nn.Linear(hidden_size, hidden_size),
    #   nn.Tanh(),
    #   nn.Linear(hidden_size, action_space),
    #   nn.Tanh(),
    # )

    # self.critic = nn.Sequential(
    #   nn.Linear(hidden_size, hidden_size),
    #   nn.Tanh(),
    #   nn.Linear(hidden_size, 1)
    # ) 

    self.actor = nn.Sequential(
      nn.Linear(state_space, 64),
      nn.Tanh(),
      nn.Linear(64, 32),
      nn.Tanh(),
      nn.Linear(32, action_space),
      nn.Tanh(),
    )

    self.critic = nn.Sequential(
      nn.Linear(state_space, 64),
      nn.Tanh(),
      nn.Linear(64, 32),
      nn.Tanh(),
      nn.Linear(32, 1)
    ) 

  def forward(self, x):
    # x = self.head(x)
    value = self.critic(x)
    actions = self.actor(x)
    return actions, value
  
  def act(self, state):
    x = torch.FloatTensor(state.reshape(1, -1))

    action_mean, value = self.forward(x)
    cov_mat = torch.diag(torch.ones(self.action_space) * 0.5**0.5)
    
    dist = MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()

    return action, dist.log_prob(action)
  
  def evaluate(self, state, action):
    action_mean, value = self.forward(state)
    cov_mat = torch.diag(torch.ones(self.action_space) * 0.5**0.5)

    dist = MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()
    
    return action, dist.log_prob(action), value, dist.entropy()