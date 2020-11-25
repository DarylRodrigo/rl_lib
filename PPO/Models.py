import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import pdb

class ActorCritic(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, state_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    
    self.head = nn.Sequential(
      nn.Linear(state_space, hidden_size),
      nn.ReLU()
    )

    self.actor = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, action_space)
    )

    self.critic = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    ) 

  def forward(self, x):
    x = self.head(x)
    value = self.critic(x)
    action = self.actor(x)
    return action, value
  
  def act(self, x):
    logits, value = self.forward(x)

    logits = F.softmax(logits, dim=-1)
    probs = Categorical(logits)
    action = probs.sample()

    return action,probs.log_prob(action)
  
  def evaluate(self, state, action):
    logits, value = self.forward(state)
    logits = F.softmax(logits, dim=-1) 
    probs = Categorical(logits)
    return action, probs.log_prob(action), value, probs.entropy()
