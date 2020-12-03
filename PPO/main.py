import functools
import gym
from Config import Config
from util import train
from Models import ActorCritic
from Networks import head_model, actor_model, critic_model
from Memory import Memory
import matplotlib.pyplot as plt


config = Config(gym.make('CartPole-v1'))
config.update_every = 50
config.num_learn = 4
config.win_condition = 400
config.n_episodes = 2000
config.max_t = 700

config.Memory = Memory
config.Model = ActorCritic
config.head_model = functools.partial(head_model, config)
config.actor_model = functools.partial(actor_model, config)
config.critic_model = functools.partial(critic_model, config)

scores, average_scores = train(config)

plt.plot(scores)
plt.plot(average_scores)
plt.show()
