import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, state_size, action_size):
    super(ActorCritic, self).__init__()

    self.head = nn.Sequential(
      nn.Linear(state_size, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU()
    )

    self.actor = nn.Sequential(
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, action_size)
    )

    self.critic = nn.Sequential(
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, x):

    x = self.head(x)

    policy = self.actor(x)
    value = self.critic(x)

    return policy, value
  
  def act(self, x):
    policy, value = self.forward(x)
    action_prob = F.softmax(policy, dim=1)

    m = Categorical(action_prob)
    action = m.sample()

    return action, m.log_prob(action), value


# ac = ActorCritic(10, 2)
# state = torch.randn(4, 10)
# action, log_prob, value =  ac.act(state)
# print(action)