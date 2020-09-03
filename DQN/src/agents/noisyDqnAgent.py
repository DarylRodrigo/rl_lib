import numpy as np
import random
from collections import namedtuple, deque

from model import NoisyQNetwork, NoisyDualingQNetwork
import pdb
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NoisyAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, seed, config={}, fc1_units=64, fc2_units=64):
    """Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      seed (int): random seed
    """
    self.BUFFER_SIZE = config["BUFFER_SIZE"] if "BUFFER_SIZE" in config else int(1e5)
    self.BATCH_SIZE = config["BATCH_SIZE"] if "BATCH_SIZE" in config else 64       
    self.GAMMA = config["GAMMA"] if "GAMMA" in config else 0.99          
    self.TAU = config["TAU"] if "TAU" in config else 1e-3            
    self.LR = config["LR"] if "LR" in config else 5e-4             
    self.UPDATE_EVERY = config["UPDATE_EVERY"] if "UPDATE_EVERY" in config else 4      
    self.LR_ANNEALING = config["LR_ANNEALING"] if "LR" in config else False
    self.DOUBLE_DQN = config["DOUBLE_DQN"] if "DOUBLE_DQN" in config else False

    print("BUFFER_SIZE: " + str(self.BUFFER_SIZE))
    print("BATCH_SIZE: " + str(self.BATCH_SIZE))
    print("GAMMA: " + str(self.GAMMA))
    print("TAU: " + str(self.TAU))
    print("LR: " + str(self.LR))
    print("UPDATE_EVERY: " + str(self.UPDATE_EVERY))
    print("DOUBLE_DQN: " + str(self.DOUBLE_DQN))

    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    self.loss = 0

    # Q-Network
    self.qnetwork_local = NoisyDualingQNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
    self.qnetwork_target = NoisyDualingQNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

    # LR scheduler - https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    if self.LR_ANNEALING:
      self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 250)
    

    # Replay memory
    self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
  
  def set_learning_rate(self, lr):
    for g in self.optimizer.param_groups:
      g['lr'] = lr
  
  def anneal_lr(self):
    if self.LR_ANNEALING:
      self.lr_scheduler.step()
  
  def get_lr(self):
    return self.optimizer.param_groups[0]['lr']
  
  def get_loss(self):
    return self.loss
  
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > self.BATCH_SIZE:
        # print("Going to learn")
        experiences = self.memory.sample()
        self.learn(experiences, self.GAMMA)
        # print("Done Learning")                
    


  def act(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """


    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    self.qnetwork_local.eval()
    with torch.no_grad():
      self.qnetwork_local.sample_noise()
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()
    

    # For array of actions
    actions = [ np.argmax(action.cpu().data.numpy()) for action in action_values[0]]
    
    return actions

  def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor

    
    Y^q = R_t+1 + gamma * Q(S_t+1, argmax_a Q(S_t+1, a; θ^target_t); θt)
    """
    states, actions, rewards, next_states, dones = experiences

    # Get expected Q values from local model
    self.qnetwork_local.sample_noise()
    Q_expected = self.qnetwork_local(states).gather(1, actions)


    # Get max action from network
    max_next_actions = self.get_max_next_actions(next_states)

    # Get what the max values are for those actions
    self.qnetwork_target.sample_noise()

    # Get max_next_q_values -> .gather("dim", "index")
    max_next_q_values = self.qnetwork_target(next_states).gather(1, max_next_actions)
    
    # Y^q
    Q_targets = rewards + (gamma * max_next_q_values * (1 - dones))

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
    self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)    
            

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

    if (self.DOUBLE_DQN):
      return self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
    else:
      return self.qnetwork_target(next_states).detach().max(1)[1].unsqueeze(1)

'''
Double Q-Learning
As per paper, we use the second set of weights to 
'''
class DoubleDqn():
  def __init__(self, state_size, action_size, seed, config={}, fc1_units=64, fc2_units=64):
    super().__init__(self, state_size, action_size, seed, config={}, fc1_units=64, fc2_units=64)
  
  def max_next_action(self, next_state):
    return self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)

class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.

    Params
    ======
      action_size (int): dimension of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)