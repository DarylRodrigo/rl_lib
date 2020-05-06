class ReplayBuffer:
  def __init__(self, buffer_size):
    self.buffer = deque(maxlen=int(buffer_size))
    self.Experience = namedtuple("experience", ["state", "action", "reward", "next_state", "done"])

  def push(self, state, action, reward, next_state, done):
    e = self.Experience(state, action, np.array([reward]), next_state ,done)
    self.buffer.append(e)

  def sample(self, batch_size):
    samples = random.sample(self.buffer, batch_size)

    states = [ exp.state for exp in samples]
    actions = [ exp.action for exp in samples]
    rewards = [ exp.reward for exp in samples]
    next_states = [ exp.next_state for exp in samples]
    dones = [ exp.done for exp in samples]

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    return  len(self.buffer)