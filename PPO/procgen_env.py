import time
import numpy as np
import torch
import gym
from procgen import ProcgenEnv
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize, VecVideoRecorder


class VecPyTorch(VecEnvWrapper):
  def __init__(self, venv, device):
    super(VecPyTorch, self).__init__(venv)
    self.device = device

  def reset(self):
    obs = self.venv.reset()
    obs = torch.from_numpy(obs).float().to(self.device)
    return obs

  def step_async(self, actions):
    actions = actions.cpu().numpy()
    self.venv.step_async(actions)

  def step_wait(self):
    obs, reward, done, info = self.venv.step_wait()
    obs = torch.from_numpy(obs).float().to(self.device)
    reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
    return obs, reward, done, info

class VecExtractDictObs(VecEnvWrapper):
  def __init__(self, venv, key):
    self.key = key
    super().__init__(venv=venv,
      observation_space=venv.observation_space.spaces[self.key])

  def reset(self):
    obs = self.venv.reset()
    return obs[self.key]

  def step_wait(self):
    obs, reward, done, info = self.venv.step_wait()
    return obs[self.key], reward, done, info


class VecMonitor(VecEnvWrapper):
  def __init__(self, venv):
    VecEnvWrapper.__init__(self, venv)
    self.eprets = None
    self.eplens = None
    self.epcount = 0
    self.tstart = time.time()

  def reset(self):
    obs = self.venv.reset()
    self.eprets = np.zeros(self.num_envs, 'f')
    self.eplens = np.zeros(self.num_envs, 'i')
    return obs

  def step_wait(self):
    obs, rews, dones, infos = self.venv.step_wait()
    self.eprets += rews
    self.eplens += 1

    newinfos = list(infos[:])
    for i in range(len(dones)):
      if dones[i]:
        info = infos[i].copy()
        ret = self.eprets[i]
        eplen = self.eplens[i]
        epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
        info['episode'] = epinfo
        self.epcount += 1
        self.eprets[i] = 0
        self.eplens[i] = 0
        newinfos[i] = info
    return obs, rews, dones, newinfos

def make_procgen_env(env_name, num_envs, device):
    venv = ProcgenEnv(env_name=env_name, num_envs=num_envs, distribution_mode="easy", num_levels=0, start_level=0)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv)
    envs = VecNormalize(venv=venv, norm_obs=False)
    envs = VecPyTorch(envs, device)
    # VecVideoRecorder(envs, f'videos/{experiment_name}', record_video_trigger=lambda x: x % 1000000== 0, video_length=100)
    return venv
