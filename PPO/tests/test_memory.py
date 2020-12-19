import torch
from ..Memory import Memory
from ..envs import make_env, VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv

def setup(num_envs=2):
    # Config
    env_id = "BreakoutNoFrameskip-v4"
    seed = 1234
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rollout_size = 100

    # Create mem
    env = VecPyTorch(DummyVecEnv([make_env(env_id, seed+i, i) for i in range(num_envs)]), device)
    mem = Memory(rollout_size, num_envs, env, device)
    return mem

def test_init_memory():
    mem = setup()
    assert True
