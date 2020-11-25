class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.log_probs = []
    self.dones = []
  
  def add(self, state, action, reward, log_prob, done):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.log_probs.append(log_prob)
    self.dones.append(done)
  
  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.log_probs = []
    self.dones = []