import copy
import gym
import torch
import numpy as np
from collections import deque
from PPO import PPO
from Config import Config
import pdb
import wandb
from envs import make_atari_env, make_env, VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv

def train(config, envs):
  scores_deque = deque(maxlen=100)
  scores = []
  average_scores = []
  global_step = 0

  agent = PPO(config)

  states = envs.reset()
  score = 0
  values, dones = None, None
  
  if config.wandb:
    wandb.watch(agent.model)

  while global_step < config.n_steps:

    while agent.mem.isFull() == False:
      global_step += config.num_env

      # Take actions
      with torch.no_grad():
        actions, log_probs, values, entrs = agent.act(states)
      next_states, rewards, dones, infos = envs.step(actions)

      # Add to memory buffer
      agent.add_to_mem(states, actions, rewards, log_probs, values, dones)

      # Update state
      states = next_states

      # Book Keeping
      for info in infos:
        if 'episode' in info:
          score = info['episode']['r']
          config.tb_logger.add_scalar("charts/episode_reward", score, global_step)
          if config.wandb:
            wandb.log({
              "episode_reward": score,
              "global_step": global_step
            })
          
          scores_deque.append(score)
          scores.append(score)
          average_scores.append(np.mean(scores_deque))

    # update and learn
    value_loss, pg_loss, approx_kl, approx_entropy, lr_now = agent.learn(config.num_learn, values, dones, global_step)
    agent.mem.reset()

    # Book Keeping
    config.tb_logger.add_scalar("losses/value_loss", value_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    config.tb_logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    config.tb_logger.add_scalar("losses/approx_entropy", approx_entropy.item(), global_step)

    if config.wandb:
      wandb.log({
        "value_loss": value_loss,
        "policy_loss": pg_loss,
        "approx_kl": approx_kl,
        "approx_entropy": approx_entropy,
        "global_step": global_step,
        "learning_rate": lr_now
       })

    print("Global Step: {}	Average Score: {:.2f}".format(global_step, np.mean(scores_deque)))   

  return scores, average_scores
