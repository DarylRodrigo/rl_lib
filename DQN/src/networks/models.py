import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def hidden_init(layer):
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)

class NoisyLinear(nn.Module):
  def __init__(self, feat_in, feat_out, std_init=0.4):
    super(NoisyLinear, self).__init__()

    self.std_init = std_init
    self.feat_in = feat_in
    self.feat_out = feat_out

    # The weights and biases (mu)
    self.weights_mu = nn.Parameter(torch.empty(feat_out, feat_in))
    self.bias_mu = nn.Parameter(torch.empty(feat_out))

    # The weights for noise to scale (sigma)
    self.weights_sigma = nn.Parameter(torch.empty(feat_out, feat_in))
    self.bias_sigma = nn.Parameter(torch.empty(feat_out))

    # Buffer to hold noise created
    self.register_buffer('weights_eps', torch.empty(feat_out, feat_in))
    self.register_buffer('bias_eps', torch.empty(feat_out))

    # Reset Parameters
    self.reset_parameters()
    self.reset_noise()

  '''
  3.2 Initialisation of Noisy Network - using factorised noisy networks
  - Each element mu is initilased from a uniform distribution U(-1/sqrt(p) , 1/sqrt(p))
  - Each element sigma is initialised to a constant sigma/sqrt(p)

  Note:
  - Uniform_(LOWER, UPPER) changes tensor values between uniform distribution
  - fill_(SET_VALUE) fills tensor with set value
  '''
  def reset_parameters(self):

    sigma_value = self.std_init / math.sqrt(self.feat_in)
    mu_range = 1 / math.sqrt(self.weights_mu.size(1))

    self.weights_mu.data.uniform_(-mu_range, mu_range)
    self.weights_sigma.data.fill_(self.std_init / math.sqrt(self.weights_sigma.size(1)))

    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

  '''
  3 NoisyNets for RL - section b
  - eps is created with f(x) = sgn(x) * sqrt(|x|)
  '''
  def _epsilon_noise(self, size):
    x = torch.randn(size)
    x = x.sign() * x.abs().sqrt()
    return x
  def sample_noise(self):
    self.reset_noise()

  def reset_noise(self):
    epsilon_in = self._epsilon_noise(self.feat_in)
    epsilon_out = self._epsilon_noise(self.feat_out)
    
    # ger does a matrix multiplication with incoming vector
    self.weights_eps.copy_(epsilon_out.ger(epsilon_in))
    self.bias_eps.copy_(self._epsilon_noise(self.feat_out))

  '''
  3 NoisyNets for RL - Formula 9
  - y = (μ_w + σ_w (dot) ε_w) * x  + (μ_b + σ_b (dot) ε_b) 

  Note:
  - We're using F.linear as we're evaluating the result not creating another linear module
  '''
  def forward(self, inp):

    # if self.training:
    #   return F.linear(inp, self.weights_mu + self.weights_sigma * self.weights_eps, self.bias_mu + self.bias_sigma * self.bias_eps)
    # else:
    #   return F.linear(inp, self.weights_mu, self.bias_mu)
    return F.linear(inp, self.weights_mu + self.weights_sigma * self.weights_eps, self.bias_mu + self.bias_sigma * self.bias_eps)


class CNNQNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed):
    # Call inheritance
    super(CNNQNetwork, self).__init__()
    self.seed = torch.manual_seed(1234)

    self.fc1 = nn.Linear(state_size, fc1_units)

    self.bn1 = nn.BatchNorm1d(fc1_units)

    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)
    self.reset_parameters()

  def reset_parameters(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class QNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    # Call inheritance
    super(QNetwork, self).__init__()
    self.seed = torch.manual_seed(1234)

    self.fc1 = nn.Linear(state_size, fc1_units)

    self.bn1 = nn.BatchNorm1d(fc1_units)

    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)
    self.reset_parameters()

  def reset_parameters(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    
    return x

class DualingQNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    # Call inheritance
    super(DualingQNetwork, self).__init__()
    self.seed = torch.manual_seed(1234)

    self.feature = nn.Sequential(
      nn.Linear(state_size, fc1_units),
      nn.ReLU()
    )

    self.advantage = nn.Sequential(
      nn.Linear(fc1_units, fc2_units),
      nn.ReLU(),
      nn.Linear(fc2_units, action_size)
    )

    self.value = nn.Sequential(
      nn.Linear(fc1_units, fc2_units),
      nn.ReLU(),
      nn.Linear(fc2_units, 1)
    )

  def forward(self, state):
    x = self.feature(state)
    
    advantage = self.advantage(x)
    value = self.value(x)

    x = value + advantage - advantage.mean()

    return x

class NoisyQNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    # Call inheritance
    super(NoisyQNetwork, self).__init__()
    self.seed = torch.manual_seed(1234)

    self.fc1 = nn.Linear(state_size, fc1_units)
    self.noisy2 = NoisyLinear(fc1_units, fc2_units)
    self.noisy3 = NoisyLinear(fc2_units, action_size)
    self.reset_parameters()

  def reset_parameters(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.noisy2.reset_parameters()
    self.noisy3.reset_parameters()
  
  def sample_noise(self):
    self.noisy2.sample_noise()
    self.noisy3.sample_noise()

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.noisy2(x))
    x = self.noisy3(x)

    return x


class NoisyDualingQNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    # Call inheritance
    super(NoisyDualingQNetwork, self).__init__()
    self.seed = torch.manual_seed(1234)

    self.feature = nn.Sequential(
      nn.Linear(state_size, fc1_units),
      nn.ReLU()
    )

    self.adv_noisy_1 = NoisyLinear(fc1_units, fc2_units)
    self.adv_noisy_2 = NoisyLinear(fc2_units, action_size)

    self.value_noisy_1 = NoisyLinear(fc1_units, fc2_units)
    self.value_noisy_2 = NoisyLinear(fc2_units, 1)

    self.reset_parameters()

  def reset_parameters(self):
    self.adv_noisy_1.reset_parameters()
    self.adv_noisy_2.reset_parameters()

    self.value_noisy_1.reset_parameters()
    self.value_noisy_2.reset_parameters()
  
  def sample_noise(self):
    self.adv_noisy_1.sample_noise()
    self.adv_noisy_2.sample_noise()

    self.value_noisy_1.sample_noise()
    self.value_noisy_2.sample_noise()
    

  def forward(self, state):
    x = self.feature(state)

    
    
    advantage = F.relu(self.adv_noisy_1(x))
    advantage = self.adv_noisy_2(advantage)

    value = F.relu(self.value_noisy_1(x))
    value = self.value_noisy_2(value)

    x = value + advantage - advantage.mean()

    return x
