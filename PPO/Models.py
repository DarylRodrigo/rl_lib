import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Networks import layer_init
import pdb 


class ActorCritic(nn.Module):
  """Some Information about ActorCritic"""
  def __init__(self, config):
    super(ActorCritic, self).__init__()

    self.head = config.head_model()

    self.actor = config.actor_model()
    self.actor.add_module(
      "actor_linear",
      layer_init(nn.Linear(config.hidden_size, config.action_space))
    )

    self.critic = config.critic_model()
    self.critic.add_module(
      "critic_linear",
      layer_init(nn.Linear(config.hidden_size, 1))
    )

  def forward(self, x):
    x = self.head(x)
    value = self.critic(x)
    action = self.actor(x)
    return action, value
  
  def act(self, x, action=None):
    logits, value = self.forward(x)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), value, probs.entropy()
