import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, config):
    super(ActorCritic, self).__init__()

    self.head = config.head_model()

    self.actor = config.actor_model()
    self.actor.add_module(
      "actor_linear",
      nn.Linear(config.hidden_size, config.action_space)
    )

    self.critic = config.critic_model()
    self.critic.add_module(
      "critic_linear",
      nn.Linear(config.hidden_size, 1)
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

    return action, probs.log_prob(action)
  
  def evaluate(self, state, action):
    logits, value = self.forward(state)
    logits = F.softmax(logits, dim=-1) 
    probs = Categorical(logits)
    return action, probs.log_prob(action), value, probs.entropy()
