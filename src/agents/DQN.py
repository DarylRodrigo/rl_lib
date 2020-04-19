import numpy as np
import random
from collections import namedtuple, deque

from ..components.memory import ReplayBuffer, PrioritiesedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pdb

class Agent():
  def __init__(self, state_size, action_size, config):
    """
    Initialize an Agent object.

    state_size (int): dimension of each state
    action_size (int): dimension of each action
    """

    if config.model is None:
      raise Exception("Please select a Model for agent")
    if config.memory is None:
       raise Exception("Please select Memory for agent") 

    self.config = config

    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(self.config.seed)

    # Q-Network
    self.qnetwork_local = self.config.model(state_size, action_size, self.seed, 64, 64).to(self.config.device)
    self.qnetwork_target = self.config.model(state_size, action_size, self.seed, 64, 64).to(self.config.device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.lr)

    # LR scheduler - https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    if self.config.lr_annealing:
      self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 250)

    # Replay memory
    self.memory = self.config.memory(config)
    
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    self.eps = self.config.eps_start
  
  def anneal_lr(self):
    if self.config.lr_annealing:
      self.lr_scheduler.step()

  def anneal_eps(self):
    self.eps = max(self.config.eps_end, self.config.eps_decay*self.eps) 
  
  # def get_lr(self):
  #     return self.optimizer.param_groups[0]['lr']
  
  # def get_loss(self):
  #     return self.loss
  
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.append_samples_to_memory(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.config.learn_every
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if self.memory.n_entries() > self.config.batch_size:
        experiences = self.memory.sample()
        self.learn(experiences, self.config.gamma)


  def act(self, state, network_only=False):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """


    state = torch.from_numpy(state).float().unsqueeze(0).to(self.config.device)

    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    
    # Noisy Agent
    if "sample_noise" in dir(self.qnetwork_local):
      print(np.argmax(action_values.cpu().data.numpy()))
      return np.argmax(action_values.cpu().data.numpy())
    # Epsilon-greedy action selections
    elif self.config.eps_greedy:
      if network_only:
        return np.argmax(action_values.cpu().data.numpy())

      # Epsilon-greedy action selection
      if random.random() > self.eps:
        return np.argmax(action_values.cpu().data.numpy())
      else:
        return random.choice(np.arange(self.action_size))
      # return [ 
      #     np.argmax(action.cpu().data.numpy()) 
      #     if random.random() > self.eps 
      #     else random.choice(np.arange(self.action_size)) 
      #     for action in action_values[0]
      # ]
    
    else:
      raise Exception("No valid exploration method selected")

  def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones, is_weights, idxs = experiences
    
    # Get expected Q values from local model
    if "sample_noise" in dir(self.qnetwork_local):
      self.qnetwork_local.sample_noise()
    Q_expected = self.qnetwork_local(states).gather(1, actions)


    # Get max action from network
    max_next_actions = self.get_max_next_actions(next_states)


    # Get max_next_q_values -> .gather("dim", "index")
    if "sample_noise" in dir(self.qnetwork_target):
      self.qnetwork_target.sample_noise()
    max_next_q_values = self.qnetwork_target(next_states).gather(1, max_next_actions)
    
    # Y^q
    Q_targets = rewards + (gamma * max_next_q_values * (1 - dones))


    if PrioritiesedReplayBuffer is self.config.memory:
      errors = Q_expected - Q_targets 
      self.memory.update_priorities(idxs, errors.detach().numpy())
    
    # pdb.set_trace()

    # Compute loss
    # - Computing loss of what the we expected the Q to be... (aka qnetwork_local(states))
    # &
    #   What the target is (reward + what the expected reward of the next action was)
    # EG
    # -> if the action was super awesome and normally did really well, but this time not... it would go down a little.
    loss = F.mse_loss(Q_expected, Q_targets)

    self.loss = loss

    # Minimize the loss
    # - Optimizer is initilaised with qnetwork_local, so will update that one.
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)                     

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
      local_model (PyTorch model): weights will be copied from
      target_model (PyTorch model): weights will be copied to
      tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

  def get_max_next_actions(self, next_states):
    '''
    Passing next_states through network will give array of all action values (in batches)
    eg: [[0.25, -0.35], [-0.74, -0.65], ...]
    - Detach will avoid gradients being calculated on the variables
    - max() gives max value in whole array (0.25)
        - max(0) gives max in dim=0 ([0.25, -0.35])
        - max(1) gives max in dim=1, therefore: two tensors, one of max values and one of index
    eg: 
        - values=tensor([0.25, -0.65, ...])
        - indices=tensor([0, 1, ...])
    - we'll want to take the [1] index as that is only the action
    eg: tensor([0, 1, ...]) 
    - unsqueeze allows us to create them into an array that is usuable for next bit (.view(-1,1) would also work)
    eg: eg: tensor([[0], [1], ...]) 
    '''

    if (self.config.double_dqn):
        return self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
    else:
        return self.qnetwork_target(next_states).detach().max(1)[1].unsqueeze(1)
  
  def append_samples_to_memory(self, states, actions, rewards, next_states, dones):
    if ReplayBuffer is self.config.memory:
      return self.memory.add(states, actions, rewards, next_states, dones) 
    
    states = torch.from_numpy(np.vstack([s for s in np.array([states])])).float().to(self.config.device)
    actions = torch.from_numpy(np.vstack([a for a in np.array([actions])])).long().to(self.config.device)
    rewards = torch.from_numpy(np.vstack([r for r in np.array([rewards])])).float().to(self.config.device)
    next_states = torch.from_numpy(np.vstack([n for n in np.array([next_states])])).float().to(self.config.device)
    dones = torch.from_numpy(np.vstack([d for d in np.array([dones])]).astype(np.uint8)).float().to(self.config.device)

    # states = torch.from_numpy(np.vstack([s for s in states])).float().to(self.config.device)
    # actions = torch.from_numpy(np.vstack([a for a in actions])).long().to(self.config.device)
    # rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().to(self.config.device)
    # next_states = torch.from_numpy(np.vstack([n for n in next_states])).float().to(self.config.device)
    # dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().to(self.config.device)

    # Get error for samples
    if "sample_noise" in dir(self.qnetwork_local):
      self.qnetwork_local.sample_noise()
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    max_next_actions = self.get_max_next_actions(next_states)
    
    if "sample_noise" in dir(self.qnetwork_target):
      self.qnetwork_target.sample_noise()
    max_next_q_values = self.qnetwork_target(next_states).gather(1, max_next_actions)
    Q_targets = rewards + (self.config.gamma * max_next_q_values * (1 - dones))

    errors = Q_expected - Q_targets 
    
    for state, action, reward, next_state, done, error in zip(states, actions, rewards, next_states, dones, errors):
      self.memory.add(
        states.detach().numpy(), 
        actions.detach().numpy(), 
        rewards.detach().numpy(), 
        next_states.detach().numpy(), 
        dones.detach().numpy(), 
        errors.detach().numpy()
      )