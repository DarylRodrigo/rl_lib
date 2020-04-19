import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)

import torch.autograd as autograd 
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    
    self.in_features  = in_features
    self.out_features = out_features
    self.std_init     = std_init
    
    self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
    
    self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
    self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
    self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
    
    self.reset_parameters()
    self.sample_noise()
  
  def forward(self, x):
    if self.training: 
      weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
      bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
    else:
      weight = self.weight_mu
      bias   = self.bias_mu
    
    return F.linear(x, weight, bias)
  
  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.weight_mu.size(1))
    
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
    
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
  
  def sample_noise(self):
    epsilon_in  = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(self._scale_noise(self.out_features))
  
  def _scale_noise(self, size):
    x = torch.randn(size)
    x = x.sign().mul(x.abs().sqrt())
    return x


# class NoisyLinear(nn.Module):
#   def __init__(self, feat_in, feat_out, std_init=0.4):
#     super(NoisyLinear, self).__init__()

#     self.std_init = std_init
#     self.feat_in = feat_in
#     self.feat_out = feat_out

#     # The weights and biases (mu)
#     self.weights_mu = nn.Parameter(torch.empty(feat_out, feat_in))
#     self.bias_mu = nn.Parameter(torch.empty(feat_out))

#     # The weights for noise to scale (sigma)
#     self.weights_sigma = nn.Parameter(torch.empty(feat_out, feat_in))
#     self.bias_sigma = nn.Parameter(torch.empty(feat_out))

#     # Buffer to hold noise created
#     self.register_buffer('weights_eps', torch.empty(feat_out, feat_in))
#     self.register_buffer('bias_eps', torch.empty(feat_out))

#     # Reset Parameters
#     self.reset_parameters()
#     self.sample_noise()

#   '''
#   3.2 Initialisation of Noisy Network - using factorised noisy networks
#   - Each element mu is initilased from a uniform distribution U(-1/sqrt(p) , 1/sqrt(p))
#   - Each element sigma is initialised to a constant sigma/sqrt(p)

#   Note:
#   - Uniform_(LOWER, UPPER) changes tensor values between uniform distribution
#   - fill_(SET_VALUE) fills tensor with set value
#   '''
#   def reset_parameters(self):
#     mu_range = 1.0 / math.sqrt(self.feat_in)
#     self.weights_mu.data.uniform_(-mu_range, mu_range)
#     self.bias_mu.data.uniform_(-mu_range, mu_range)

#     sigma_value = self.std_init / math.sqrt(self.feat_in)
#     self.weights_sigma.data.fill_(sigma_value)
#     self.bias_sigma.data.fill_(sigma_value)

#   '''
#   3 NoisyNets for RL - section b
#   - eps is created with f(x) = sgn(x) * sqrt(|x|)
#   '''
#   def _epsilon_noise(self, size):
#     x = torch.randn(size)
#     x = x.sign() * x.abs().sqrt()
#     return x

#   def sample_noise(self):
#     epsilon_in = self._epsilon_noise(self.feat_in)
#     epsilon_out = self._epsilon_noise(self.feat_out)
    
#     # ger does a matrix multiplication with incoming vector
#     self.weights_eps.copy_(epsilon_out.ger(epsilon_in))
#     self.bias_eps.copy_(self._epsilon_noise(self.feat_out))

#   '''
#   3 NoisyNets for RL - Formula 9
#   - y = (μ_w + σ_w (dot) ε_w) * x  + (μ_b + σ_b (dot) ε_b) 

#   Note:
#   - We're using F.linear as we're evaluating the result not creating another linear module
#   '''
#   def forward(self, inp):

#     # if self.training:
#     #   return F.linear(inp, self.weights_mu + self.weights_sigma * self.weights_eps, self.bias_mu + self.bias_sigma * self.bias_eps)
#     # else:
#     #   return F.linear(inp, self.weights_mu, self.bias_mu)
#     return F.linear(inp, self.weights_mu + self.weights_sigma * self.weights_eps, self.bias_mu + self.bias_sigma * self.bias_eps)

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
    # x = F.relu(self.bn1(self.fc1(state)))
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    # return F.tanh(x)
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
    
    advantage = nn.ReLU(self.adv_noisy_1(x))
    advantage = self.adv_noisy_2(x)

    value = nn.ReLU(self.value_noisy_1(x))
    value = self.value_noisy_2(x)

    x = value + advantage - advantage.mean()

    return x

