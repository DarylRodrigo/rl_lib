from ActorCritic import ActorCritic
from ActorCriticContinuous import ActorCriticContinuous
from Memory import Memory
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO:
  def __init__(self, state_space, action_space, hidden_size=64, epsilon=0.2, entropy_beta=0.01, gamma=0.99, lr=0.002):
    self.mem = Memory()

    self.gamma = gamma
    self.epsilon = epsilon
    self.entropy_beta = entropy_beta

    self.model = ActorCritic(state_space, action_space, hidden_size).to(device)
    self.model_old = ActorCritic(state_space, action_space, hidden_size).to(device)

    self.model_old.load_state_dict(self.model.state_dict())

    self.optimiser = optim.Adam(self.model.parameters(), lr=lr)
  
  def act(self, x):
    return self.model_old.act(x)

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
    discounted_returns = torch.FloatTensor(discounted_returns).to(device)
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

    prev_states = torch.stack(self.mem.states).to(device).detach()
    prev_actions = torch.stack(self.mem.actions).to(device).detach()
    prev_log_probs = torch.stack(self.mem.log_probs).to(device).detach()

    for i in range(num_learn):

      # find ratios
      actions, log_probs, values, entropy = self.model.evaluate(prev_states, prev_actions)
      ratio = torch.exp(log_probs - prev_log_probs.detach())

      # calculate advantage
      advantage = discounted_returns - values.cpu().detach()

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
  
  def learn_gae(self, num_learn, last_value, last_done):
    gae_lamda = 1
    for i in range(num_learn):

      # Get state memory
      prev_states = torch.stack(self.mem.states).to(device).detach()
      prev_actions = torch.stack(self.mem.actions).to(device).detach()
      prev_log_probs = torch.stack(self.mem.log_probs).to(device).detach()
      prev_rewards = self.mem.rewards
      prev_dones = self.mem.dones

      # find ratios
      actions, log_probs, values, entropy = self.model.evaluate(prev_states, prev_actions)
      ratio = torch.exp(log_probs - prev_log_probs.detach())

      # Calculate discounted rewards
      advantage = []
      last_gae_lam=0

      for step in reversed(range(len(self.mem.rewards))):
        
        if step == len(self.mem.rewards) - 1:
          next_value = last_value
          next_non_terminal = 1.0 - last_done
        else:
          next_value = values[step+1]
          next_non_terminal = 1.0 - prev_dones[step + 1]

        delta = prev_rewards[step] + (self.gamma  * next_value * prev_dones[step]) - values[step]
        last_gae_lam = delta + (self.gamma * gae_lamda * next_non_terminal * last_gae_lam)

        advantage.insert(0,last_gae_lam)


      # normalise advantage
      # # advantages = torch.FloatTensor(advantages).to(device)
      # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

      # calculate discounted returns
      # pdb.set_trace()
      discounted_returns = advantage + values
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

class PPOContinuous(PPO):
  def __init__(self, state_space, action_space, hidden_size=64, epsilon=0.2, entropy_beta=0.01, gamma=0.99, lr=0.002):
    self.mem = Memory()

    self.gamma = gamma
    self.epsilon = epsilon
    self.entropy_beta = entropy_beta

    self.model = ActorCriticContinuous(state_space, action_space, hidden_size).to(device)
    self.model_old = ActorCriticContinuous(state_space, action_space, hidden_size).to(device)

    self.model_old.load_state_dict(self.model.state_dict())

    self.optimiser = optim.Adam(self.model.parameters(), lr=lr)
  
  def act(self, x):
    return self.model_old.act(x)

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
    discounted_returns = torch.FloatTensor(discounted_returns).to(device)
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

    

    prev_states = torch.stack(self.mem.states).to(device).detach()
    prev_actions = torch.FloatTensor(self.mem.actions).to(device).detach()
    prev_log_probs = torch.FloatTensor(self.mem.log_probs).to(device).detach()

    for i in range(num_learn):

      # find ratios
      actions, log_probs, values, entropy = self.model.evaluate(prev_states, prev_actions)
      ratio = torch.exp(log_probs - prev_log_probs.detach())
    
      # calculate advantage
      advantage = discounted_returns - values.detach()

      # calculate surrogates
      surrogate_1 = ratio * advantage
      surrogate_2 = torch.clamp(advantage, 1-self.epsilon, 1+self.epsilon)
      loss = -torch.min(surrogate_1, surrogate_2) + 0.5*F.mse_loss(values, discounted_returns) - self.entropy_beta*entropy

      loss = loss.mean()

      # calculate gradient
      self.optimiser.zero_grad()
      loss.backward()
      self.optimiser.step()
    
    self.model_old.load_state_dict(self.model.state_dict())
