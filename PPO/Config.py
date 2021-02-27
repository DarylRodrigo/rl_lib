from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import wandb
import time
import torch
from envs import make_env, make_atari_env, VecPyTorch
from procgen_env import make_procgen_env
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import numpy as np

class Config:
  def __init__(self, env_id, env_type="gym", num_envs=8):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running experiment {} -  on device: {}".format(env_id, self.device))
    self.seed = 1
    self.num_env = num_envs

    self.env_id = env_id
    if env_type == "atari":
      self.env  = VecPyTorch(DummyVecEnv([make_atari_env(self.env_id, self.seed+i, i) for i in range(self.num_env)]), self.device)
      self.state_space = self.env.observation_space.shape[0]
      self.action_space = self.env.action_space.n
      self.channels = 4
    elif env_type == "procgen":
      self.env = env = make_procgen_env(env_id, self.num_env, self.device)
      self.state_space = self.env.observation_space.shape[0]
      self.action_space = self.env.action_space.n
      self.channels = 3
    elif env_type == "gym":
      self.env = VecPyTorch(DummyVecEnv([make_env(self.env_id, self.seed+i, i) for i in range(self.num_env)]), self.device)
      self.state_space = self.env.observation_space.shape[0]
      self.action_space = self.env.action_space.n

    self.win_condition = None

    self.n_steps = 7000000
    self.n_episodes = 2000
    self.max_t = 100
    self.update_every = 100

    self.epsilon = 0.1
    self.eps_start = 1.0
    self.eps_end = 0.01
    self.eps_decay = 0.995

    self.gamma = 0.99
    self.lr = 1e-5
    self.hidden_size = 64

    self.memory = None
    self.mini_batch_size = 256
    self.gae = True
    self.gae_lambda = 0.95
    self.lr_annealing = False
    self.epsilon_annealing = False
    self.learn_every = 4
    self.entropy_beta = 0.01

    self.model = None

    self.wandb = False
    self.save_loc = None
    
    # Set up logging for tensor board
    experiment_name = f"{env_id}____{int(time.time())}"
    self.tb_logger = SummaryWriter(f"runs/{experiment_name}")

    self.init_seed()
  
  def init_seed(self):
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)

  def print_config(self):
    pprint(vars(self))
  
  def init_wandb(self, project="rl-lib", entity="procgen"):
    wandb.init(project=project, entity=entity, sync_tensorboard=True)

    wandb.config.update({
      "env_id": self.env_id,
      "seed": self.seed,
      "n_episodes": self.n_episodes,
      "max_t": self.max_t,
      "epsilon": self.epsilon,
      "eps_start": self.eps_start,
      "eps_end": self.eps_end,
      "eps_decay": self.eps_decay,
      "gamma": self.gamma,
      "lr": self.lr,
      "hidden_size": self.hidden_size,
      "mini_batch_size": self.mini_batch_size,
      "lr_annealing": self.lr_annealing,
      "learn_every": self.learn_every,
      "entropy_beta": self.entropy_beta,
      "update_every": self.update_every,
      "num_learn": self.num_learn
    })

    self.wandb = True

