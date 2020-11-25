import gym
from Config import Config
from util import train
from Models import ActorCritic
from Memory import Memory


config = Config(gym.make('CartPole-v1'))
config.update_every = 2000
config.num_learn = 4
config.win_condition = 230
config.n_episodes = 4000
config.max_t = 700


config.Memory = Memory
config.Model = ActorCritic


scores = train(config)
