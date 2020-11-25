import torch

class Config:
  def __init__(self, env):
    self.env = env
    self.state_space = env.observation_space.shape[0]
    self.action_space = env.action_space.n

    self.win_condition = None

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.seed = 123456789

    self.n_episodes = 2000
    self.max_t = 1000
    # TODO set as decaying and pass into learn from PPO
    self.epsilon = 0.2
    self.eps_start = 1.0
    self.eps_end = 0.01
    self.eps_decay = 0.995

    self.eps_greedy = True
    self.noisy = False

    self.tau = 1e-3 
    self.gamma = 0.99
    self.lr = 5e-4

    self.memory = None
    self.batch_size = 64
    self.buffer_size = int(1e5)
    self.lr_annealing = False
    self.learn_every = 4
    self.entropy_beta = 0.01

    self.double_dqn = False
    self.model = None

    self.save_loc = None
  
  def print_config(self):
    print("Agent Configuration:")
    print("env: \t\t{}".format(self.env.spec))
    # print("state space\t{}".format(self.env.observation_space.shape))
    # print("action space\t{}".format(self.env.action_space.n))

    print("win condition: \t{}".format(self.win_condition))
    print("device: \t{}".format(self.device))
    print("seed: \t\t{}".format(self.seed))

    print("n_episodes: \t{}".format(self.n_episodes))
    print("max_t: \t\t{}".format(self.max_t))
    print("eps_start: \t{}".format(self.eps_start))
    print("eps_end: \t{}".format(self.eps_end))
    print("eps_decay: \t{}".format(self.eps_decay))

    print("eps_greedy: \t{}".format(self.eps_greedy))
    print("noisy: \t\t{}".format(self.noisy))


    print("tau: \t\t{}".format(self.tau))
    print("gamma: \t\t{}".format(self.gamma))
    print("lr: \t\t{}".format(self.lr))

    print("memory: \t{}".format(self.memory))
    print("batch_size: \t{}".format(self.batch_size))
    print("buffer_size: \t{}".format(self.buffer_size))
    print("lr_annealing: \t{}".format(self.lr_annealing))
    print("learn_every: \t{}".format(self.learn_every))

    print("double_dqn: \t{}".format(self.double_dqn))
    print("model: \t\t{}".format(self.model))

    print("save_loc: \t{}".format(self.save_loc))
