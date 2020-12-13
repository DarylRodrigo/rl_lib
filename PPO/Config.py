from pprint import pprint
import torch

class Config:
  def __init__(self, env):
    self.env = env
    self.state_space = env.observation_space.shape[0]
    self.action_space = env.action_space.n

    self.win_condition = None

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running experiment with device: {}".format(self.device))
    self.seed = 123456789

    self.total_global_steps = 10e6
    self.n_episodes = 2000
    self.max_t = 1000
    # TODO set as decaying and pass into learn from PPO
    self.epsilon = 0.1
    self.eps_start = 1.0
    self.eps_end = 0.01
    self.eps_decay = 0.995

    self.gamma = 0.99
    self.lr = 5e-4
    self.hidden_size = 64

    self.memory = None
    self.batch_size = 64
    self.buffer_size = int(1e5)
    self.lr_annealing = False
    self.epsilon_annealing = False
    self.learn_every = 4
    self.entropy_beta = 0.01

    self.model = None

    self.save_loc = None
  
  def print_config(self):
    pprint(vars(self))
