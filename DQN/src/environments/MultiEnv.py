# https://www.tutorialspoint.com/multiprocessing-in-python

from multiprocessing import Process, Pipe
from collections import namedtuple
from time import sleep
import numpy as np
import gym

def worker(env_name, conn, idx):
  proc_running = True
  env = gym.make(env_name)
  env.reset()

  while proc_running:
    cmd, msg = conn.recv()

    if (cmd == "step"):
      next_state, reward, done, _ = env.step(msg)
      if done:
        next_state = env.reset()
      conn.send((next_state, reward, done, _))

    elif (cmd == "reset"):
      next_state = env.reset() 
      conn.send(next_state) 

    elif (cmd == "close"):
      proc_running = False
      conn.close()

    else:
      raise Exception("Command not implemented")

class MultiEnv:
  def __init__(self, env_name, num_envs):

    self.num_envs = num_envs
    self.process = namedtuple("Process", field_names=["proc", "connection"])

    self.procs = []

    for idx in range(num_envs):
      parent_conn, worker_conn = Pipe()
      proc = Process(target=worker, args=(env_name,worker_conn, idx))
      proc.start()

      self.procs.append(self.process(proc, parent_conn))

  def reset(self):
    [ p.connection.send(("reset", "")) for p in self.procs] 
    res = [ p.connection.recv() for p in self.procs]

    return np.vstack(res) 

  def step(self, actions):
    
    # send actions to envs
    [ p.connection.send(("step", action)) for p, action in zip(self.procs, actions)]
    
    # Receive response from envs.
    res = [ p.connection.recv() for p in self.procs]
    next_states, rewards, dones, _ = zip(*res)

    return np.vstack(next_states), np.array(rewards), np.array(dones), np.array(_)
  
  def close(self):
    [ p.connection.send(("close", "")) for p in self.procs] 

