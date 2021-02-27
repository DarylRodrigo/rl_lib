import functools
import gym
from Config import Config
from util import train
from Models import ActorCritic, ActorCriticCnn, ActorCriticCnnProcGen
from Networks import cnn_head_model, actor_model, critic_model, head_model
from Memory import Memory
from envs import make_atari_env, make_env, VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv

import matplotlib.pyplot as plt
import numpy as np
import pdb


def runProcGen():
  env_id = "coinrun"
  config = Config(env_id, env_type="procgen", num_envs=64)

  config.update_every = 256
  config.num_learn = 3
  config.win_condition = 230
  config.n_steps = 1e8
  config.hidden_size = 512
  config.lr = 2.5e-4
  config.lr_annealing = True
  config.epsilon_annealing = True

  config.memory = Memory
  config.model = ActorCriticCnnProcGen

  config.init_wandb()

  scores, average_scores = train(config, config.env) 

def runAtari():
  env_id = "BreakoutNoFrameskip-v4"
  config = Config(env_id, env_type="atari")

  config.update_every = 128
  config.num_learn = 4
  config.win_condition = 230
  config.n_steps = 7e6
  config.hidden_size = 512
  config.lr = 2.5e-4
  config.lr_annealing = True
  config.epsilon_annealing = True

  config.memory = Memory
  config.model = ActorCriticCnn

  config.init_wandb()

  scores, average_scores = train(config, config.env)

def runGym():
  env_id = "CartPole-v1"
  config = Config(env_id, env_type="gym")

  config.update_every = 256*4
  config.num_learn = 20
  config.win_condition = 200
  config.n_steps = 2.5e5
  config.hidden_size = 256
  config.lr = 0.002
  config.lr_annealing = True
  config.epsilon_annealing = True

  config.memory = Memory
  config.model = ActorCritic

  config.init_wandb(project="gym", entity="procgen")

  scores, average_scores = train(config, config.env)

def runLunarLander():
  env_id = "LunarLander-v2"
  
  config = Config(env_id, env_type="gym")

  config.update_every = 256*4
  config.num_learn = 20
  config.win_condition = 200
  config.n_steps = 1e7
  config.hidden_size = 256
  config.lr = 0.002
  config.lr_annealing = True
  config.epsilon_annealing = True

  config.memory = Memory    
  config.model = ActorCritic

  config.init_wandb(project="gym", entity="procgen")

  scores, average_scores = train(config, config.env)


