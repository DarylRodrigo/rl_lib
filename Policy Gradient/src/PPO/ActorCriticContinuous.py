import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import matplotlib.pyplot as plt

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCriticContinuous(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, state_space, action_space, hidden_size):
    super(ActorCriticContinuous, self).__init__()

    self.action_space = action_space
    self.state_space = state_space
    
    self.actor = nn.Sequential(
      nn.Linear(state_space, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, action_space),
      nn.Tanh(),
    )

    self.critic = nn.Sequential(
      nn.Linear(state_space, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, 1)
    ) 

  def forward(self, x):
    value = self.critic(x)
    actions = self.actor(x)
    return actions, value
  
  def act(self, x):
    action_mean, value = self.forward(x)
    cov_mat = torch.diag(torch.ones(self.action_space).to(device) * 0.5**0.5)
    
    dist = MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()

    return action, dist.log_prob(action)
  
  def evaluate(self, state, action):
    action_mean, value = self.forward(state)
    cov_mat = torch.diag(torch.ones(self.action_space).to(device) * 0.5**0.5)

    dist = MultivariateNormal(action_mean, cov_mat)
    
    return action, dist.log_prob(action), value, dist.entropy()
    

class ActorCriticLSTMContinuous(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, state_space, action_space, hidden_size):
    super(ActorCriticLSTMContinuous, self).__init__()

    self.action_space = action_space
    self.state_space = state_space

    self.num_layers = 6
    self.lstm_input_size = hidden_size
    self.hidden_size = hidden_size

    self.head = nn.Sequential(
      nn.Linear(state_space, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, hidden_size),
      nn.Tanh(),
    )
    
    self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, self.num_layers)
    
    self.actor = nn.Sequential(
      nn.Linear(hidden_size, action_space),
      nn.Tanh(),
    )

    self.critic = nn.Sequential(
      nn.Tanh(),
      nn.Linear(hidden_size, 1)
    ) 

  def forward(self, x):
    x = self.head(x)
    x = x.unsqueeze(dim=1)

    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0, c0))
    x = out[:, -1, :]

    value = self.critic(x)
    actions = self.actor(x)
    return actions, value
  
  def act(self, x):
    action_mean, value = self.forward(x)
    cov_mat = torch.diag(torch.ones(self.action_space).to(device) * 0.5**0.5)
    
    dist = MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()

    return action, dist.log_prob(action)
  
  def evaluate(self, state, action):
    action_mean, value = self.forward(state)
    cov_mat = torch.diag(torch.ones(self.action_space).to(device) * 0.5**0.5)

    dist = MultivariateNormal(action_mean, cov_mat)
    
    return action, dist.log_prob(action), value, dist.entropy()

# model = ActorCriticLSTMContinuous(8, 4, 64)
# inp = torch.rand(1, 8)
# actions, value = model.forward(inp)

# print(actions)
# print(value)