import torch
from envs import make_env, VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv

# env = config.env
env = VecPyTorch(DummyVecEnv([make_env("BreakoutNoFrameskip-v4", 234+i, i) for i in range(8)]), "cpu")

# ### TEST
# states = envs.reset()
# actions = [envs.action_space.sample() for _ in range(8)]
# actions = torch.IntTensor(actions)
# next_states, rewards, dones, _ = envs.step(actions)
# print(torch.equal(states, next_states))
# ### END TEST


states = env.reset()
actions = [env.action_space.sample() for _ in range(8)]
actions = torch.IntTensor(actions)
s, _, _, _ = env.step(actions)

print(torch.equal(s, states))
