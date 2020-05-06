import numpy as np
import random

from OUNoise import OUNoise 
from Memory import ReplayBuffer 
from Network import Actor, Critic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class DDPG:
  def __init__(self, state_space, action_space):
    self.actor = Actor(state_space, action_space).to(device)
    self.critic = Critic(state_space, action_space).to(device)

    self.actor_target = Actor(state_space, action_space).to(device)
    self.critic_target = Critic(state_space, action_space).to(device)

    self.actor_optimiser = optim.Adam(actor.parameters(), lr=1e-3)
    self.critic_optimiser = optim.Adam(critic.parameters(), lr=1e-3)

    self.mem = ReplayBuffer(buffer_size)
  
  def act(self, state, add_noise=False):
    return self.actor.act(state, add_noise)

  def save(self, fn):
    torch.save(self.actor.state_dict(), "{}_actor_model.pth".format(fn))
    torch.save(self.critic.state_dict(), "{}_critic_model.pth".format(fn))
  
  def learn(self):

    state_batch, action_batch, reward_batch, next_state_batch, masks = self.mem.sample(batch_size)
    
    state_batch = torch.FloatTensor(state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    masks = torch.FloatTensor(masks).to(device)

    # Update Critic
    self.update_critic(
        states=state_batch,
        next_states=next_state_batch,
        actions=action_batch,
        rewards=reward_batch,
        dones=masks
    )
    
    # Update actor
    self.update_actor(states=state_batch)
    
    # Update target networks
    self.update_target_networks()
  
  def update_actor(self, states):
    actions_pred = self.actor(states)  
    loss = -self.critic(states, actions_pred).mean()
    
    self.actor_optimiser.zero_grad()
    loss.backward()
    self.actor_optimiser.step()
  
  def update_critic(self, states, next_states, actions, rewards, dones):
    next_actions = self.actor_target.forward(next_states)
    
    y_i =  rewards + ( gamma * self.critic_target(next_states, next_actions) * (1-dones ))
    expected_Q = self.critic(states, actions)

    loss = F.mse_loss(y_i, expected_Q)
    
    self.critic_optimiser.zero_grad()
    loss.backward()
    self.critic_optimiser.step()
  
  def update_target_networks(self):
    for target, local in zip(self.actor_target.parameters(), self.actor.parameters()):
        target.data.copy_(tau*local.data + (1.0-tau)*target.data)
        
    for target, local in zip(self.critic_target.parameters(), self.critic.parameters()):
        target.data.copy_(tau*local.data + (1.0-tau)*target.data)