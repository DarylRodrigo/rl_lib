import functools
import gym
from Config import Config
from util import train_pixel
from Models import ActorCritic
from Networks import cnn_head_model, actor_model, critic_model, head_model
from Memory import Memory
from baselines.common.atari_wrappers import wrap_deepmind, make_atari

import matplotlib.pyplot as plt
import numpy as np

env_id = "BreakoutNoFrameskip-v4"
config = Config(env_id)

config.update_every = 128
config.num_learn = 4
config.win_condition = 230
config.n_steps = 7e6
config.hidden_size = 512
config.lr = 2.5e-4
config.lr_annealing = True
config.epsilon_annealing = True

config.memory = Memory
config.model = ActorCritic
config.head_model = functools.partial(cnn_head_model, config)
config.actor_model = functools.partial(actor_model, config)
config.critic_model = functools.partial(critic_model, config)

# config.init_wandb()

scores = train_pixel(config)
