import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import copy
import pdb
import numpy as np

class PPOBase:
  def __init__(self, config):
    self.mem = config.memory(config.update_every, config.num_env, config.env, config.device)

    self.gamma = config.gamma
    self.epsilon = config.epsilon
    self.entropy_beta = config.entropy_beta
    self.device = config.device

    self.model = config.model(config).to(self.device)
    self.model_old = config.model(config).to(self.device)

    self.model_old.load_state_dict(self.model.state_dict())

    self.optimiser = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
  
  def act(self, x):
    raise NotImplemented
  
  def add_to_mem(self, state, action, reward, log_prob, done):
    raise NotImplemented

  def learn(self, num_learn):
    raise NotImplemented

class PPOClassical(PPOBase):
  def __init__(self, config):
    super(PPOClassical, self).__init__(config)
  
  def act(self, x):
    x = torch.FloatTensor(x)
    return self.model_old.act(x)
  
  def add_to_mem(self, state, action, reward, log_prob, done):
    state = torch.FloatTensor(state)
    self.mem.add(state, action, reward, log_prob, done)

  def learn(self, num_learn):
    # Calculate discounted rewards
    discounted_returns = []
    running_reward = 0

    for reward, done in zip(reversed(self.mem.rewards), reversed(self.mem.dones)):
      if done:
        running_reward = 0
      running_reward = reward + self.gamma * running_reward

      discounted_returns.insert(0,running_reward)

    # normalise rewards
    discounted_returns = torch.FloatTensor(discounted_returns).to(self.device)
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

    prev_states = torch.stack(self.mem.states).to(self.device).detach()
    prev_actions = torch.stack(self.mem.actions).to(self.device).detach()
    prev_log_probs = torch.stack(self.mem.log_probs).to(self.device).detach()

    for i in range(num_learn):

      # find ratios
      actions, log_probs, values, entropy = self.model.act(prev_states, prev_actions)
      ratio = torch.exp(log_probs - prev_log_probs.detach())

      # calculate advantage
      advantage = discounted_returns - values

      # TODO: normalise advantages

      # calculate surrogates
      surrogate_1 = ratio * advantage
      surrogate_2 = torch.clamp(advantage, 1-self.epsilon, 1+self.epsilon)
      loss = -torch.min(surrogate_1, surrogate_2) + F.mse_loss(values, discounted_returns) - self.entropy_beta*entropy

      loss = loss.mean()

      # calculate gradient
      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()
    
    self.model_old.load_state_dict(self.model.state_dict())

class PPOPixel(PPOBase):
  def __init__(self, config):
    super(PPOPixel, self).__init__(config)
    self.config = config
  
  
  def add_to_mem(self, state, action, reward, log_prob, done):
    self.mem.add(state, action, reward, log_prob, done)
  
  def act(self, x):
    x = x.to(self.config.device)
    return self.model_old.act(x)

  def learn(self, num_learn, last_value, next_done):
    # Calculate discounted returns using rewards collected from environments
    self.mem.calculate_discounted_returns(last_value, next_done)
    
    for i in range(num_learn):
      # itterate over mini_batches
      for mini_batch_idx in self.mem.get_mini_batch_idxs(mini_batch_size=100):

        # Grab sample from memory
        prev_states, prev_actions, prev_log_probs, discounted_returns = self.mem.sample(mini_batch_idx)

        # find ratios
        actions, log_probs, _, entropy = self.model.act(prev_states, prev_actions)
        ratio = torch.exp(log_probs - prev_log_probs.detach())
        
        values = self.model_old.get_values(prev_states).reshape(-1)

        # Stats
        approx_kl = (prev_log_probs - log_probs).mean()

        # calculate advantage & normalise
        advantage = discounted_returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # calculate surrogates
        surrogate_1 = advantage * ratio
        surrogate_2 = advantage * torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)

        new_values = self.model.get_values(prev_states).view(-1)

        # Calculate losses
        value_loss =  (discounted_returns - values).pow(2).mean()
        pg_loss = -torch.min(surrogate_1, surrogate_2).mean()
        entropy_loss = entropy.mean()

        loss = pg_loss + value_loss - self.entropy_beta*entropy_loss

        # calculate gradient
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimiser.step()

        if torch.abs(approx_kl) > 0.03:
          break
      
      self.model_old.load_state_dict(self.model.state_dict())

    return value_loss, pg_loss, approx_kl, entropy_loss
