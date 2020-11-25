import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import pdb

class PPO:
  # def __init__(self, state_space, action_space, hidden_size=64, epsilon=0.2, entropy_beta=0.01, gamma=0.99, lr=0.002):
  def __init__(self, config):
    self.mem = config.Memory()

    self.gamma = config.gamma
    self.epsilon = config.epsilon
    self.entropy_beta = config.entropy_beta
    self.device = config.device

    hidden_size = 64

    self.model = config.Model(config.state_space, config.action_space, hidden_size).to(self.device)
    self.model_old = config.Model(config.state_space, config.action_space, hidden_size).to(self.device)

    self.model_old.load_state_dict(self.model.state_dict())

    self.optimiser = optim.Adam(self.model.parameters(), lr=config.lr)
  
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
    discounted_returns = torch.FloatTensor(discounted_returns).to(self.device)
    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

    prev_states = torch.stack(self.mem.states).to(self.device).detach()
    prev_actions = torch.stack(self.mem.actions).to(self.device).detach()
    prev_log_probs = torch.stack(self.mem.log_probs).to(self.device).detach()

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
