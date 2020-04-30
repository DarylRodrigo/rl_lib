import numpy as np
import random
from collections import namedtuple, deque
import torch
import time
import pdb
from .SumTree import SumTree
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, config):
    """Initialize a ReplayBuffer object.

    Params
    ======
      action_size (int): dimension of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """

    self.memory = deque(maxlen=config.buffer_size)  
    self.batch_size = config.batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(config.seed)
    self.device = config.device
  
  def add(self, state, action, reward, next_state, done, error=0):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    weights = torch.ones(len(experiences)).float().to(self.device)
    memory_loc = torch.zeros(len(experiences)).float().to(self.device)

    return (states, actions, rewards, next_states, dones, weights, memory_loc)

  def n_entries(self):
    return len(self.memory)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)

class NaivePrioritiesedReplayBuffer:
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
    self.buffer_size = buffer_size

    self.memory = deque(maxlen=buffer_size)  
    self.error = deque(maxlen=buffer_size)

    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)

  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    
    self.memory.append(e)

    # Set error to infinity; this will ensure it has the highest priority of being selected.
    self.error.append(np.inf)
  
  def n_entries(self):
    return len(self.memory)
  
  def sample(self, beta=0.5):
    start_time = time.time()

    if (len(self.memory) < self.batch_size):
        raise Exception("Not enough samples in memory to fetch this batch size")

    # Calculate Sampling Priorities
    prio = self._calculate_sampling_priorities()

    # Get array of ranks
    # - lowest first - therefore need to reverse (flip)
    # - eg in array [3,6,8,1] -> [3, 0, 1, 2] (as 1 is the lowest at index 3)
    rank_idx = np.flip(np.argsort(prio))

    # Rank Probabilities (rank 1 = 1/1, rank 2 = 1/2, rank 3 = 1/3)
    # Probabilities must sum to 1
    rank_probabilities = [1/(x+1) for x in range(len(prio))]
    rank_probabilities_sum_to_one = rank_probabilities / np.sum(rank_probabilities)

    # The rank numbers selected (eg [3, 6, 1]) -> rank 2nd Rank, 5th Rank, and 2nd Rank
    # numpy.random.choice("range", "size", probabilities)
    # - Note, this will mean that there might duplicated experiences in a batch
    sampled_rank_idx =  np.random.choice(len(self.memory), self.batch_size, p=rank_probabilities_sum_to_one)
    loc_in_buffer = rank_idx[sampled_rank_idx]

    middle_time = time.time() - start_time

    
    '''
    Fetch prioritiesed in experiences from memory 
    
    NOTE: In theory this is a nice way of doing it, but very computationally heavy 
          the more experiences in the memory the longer it takes to convert it to and do a look up.
    '''
    
    experiences = np.array(self.memory)[loc_in_buffer]
    experiences = [self.experience(e[0],e[1],e[2],e[3],e[4]) for e in experiences]

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    # Update weightings
    is_weights = ((1 / len(prio)) * (1 /  np.array(rank_probabilities)[loc_in_buffer]))**beta
    is_weights = torch.from_numpy(is_weights).float().to(device)

    return (states, actions, rewards, next_states, dones, is_weights, loc_in_buffer)

  def update_priorities(self, loc_in_buffer, err):
    updated_error = np.array(self.error)
    updated_error[loc_in_buffer] = np.array(err)
    self.error = deque(updated_error ,maxlen=self.buffer_size)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)

  '''
  Stochastic Prioritiesation - formula 3
  P(i) = p^α_i / SUM_all(p^α_k)
  '''
  def _calculate_sampling_priorities(self, offset=0.1, alpha=0.7):
    p_t = (np.absolute(np.array(self.error)) + offset)**alpha
    priorities = p_t / p_t.sum()

    return priorities
  
class PrioritiesedReplayBuffer:
  def __init__(self, config):
    self.batch_size = config.batch_size

    self.tree = SumTree(config.buffer_size)

    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = config.seed
    self.device = config.device

  def add(self, state, action, reward, next_state, done, error):
    experience = self.experience(state, action, reward, next_state, done)
    priority = self._calculate_sampling_priorities(error[0][0])

    if type(priority) == np.float64:
      self.tree.add(priority, experience)
    else:
      pdb.set_trace()
  
  def n_entries(self):
    return self.tree.n_entries
  
  def sample(self, beta=0.4):
    experiences  = []
    idxs = []
    priorities = []

    for i in range(self.batch_size):
      p = random.uniform(0, self.tree.total())
      idx, priority, experience = self.tree.get(p)

      idxs.append(idx)
      priorities.append(priority)
      experiences.append(experience)

    try:
      # Importance sampling as per Algorithm 1 - line 10.
      sampling_probabilities = priorities / self.tree.total()
      is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
      is_weights /= is_weights.max()
    except:
      pdb.set_trace()
    
    # Create into torch tensors
    try:
      states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not 0])).float().to(self.device)
      actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not 0])).long().to(self.device)
      rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not 0])).float().to(self.device)
      next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not 0])).float().to(self.device)
      dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not 0]).astype(np.uint8)).float().to(self.device) 
      is_weights = torch.from_numpy(is_weights).float().to(self.device)
    except:
      pdb.set_trace()

    return (states, actions, rewards, next_states, dones, is_weights, idxs)

  def update_priorities(self, idxs, errors):
    for idx, error in zip(idxs, errors):
      priority = self._calculate_sampling_priorities(error[0])
      if type(priority) == np.float64:
        self.tree.update(idx, priority)
      else:
        pdb.set_trace()
  
      
  
  '''
  Stochastic Prioritiesation - formula 3
  P(i) = p^α_i / SUM_all(p^α_k)
  '''
  def _calculate_sampling_priorities(self, error, offset=0.1, alpha=0.6):
    p_t = (np.abs(error) + offset)**alpha
    return p_t