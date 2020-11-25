import gym
from Config import Config
from util import train


config = Config(gym.make('CartPole-v1'))
config.update_every = 2000
config.num_learn = 4
config.win_condition = 230
config.n_episodes = 4000
config.max_t = 700


scores = train(config)
