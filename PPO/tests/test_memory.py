import torch
from ..Memory import Memory
from ..envs import make_env, VecPyTorch
from stable_baselines3.common.vec_env import DummyVecEnv

def setup(rollout_size=100, num_envs=2):
    # Config
    env_id = "BreakoutNoFrameskip-v4"
    seed = 1234
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create mem
    env = VecPyTorch(DummyVecEnv([make_env(env_id, seed+i, i) for i in range(num_envs)]), device)
    mem = Memory(rollout_size, num_envs, env, device)
    return mem, env

def test_init_memory():
    mem, env = setup()
    assert True

def test_add():
    rollout_size = 10
    num_envs = 2
    mem, env = setup(rollout_size, num_envs)
    env.reset()

    actions = [env.action_space.sample() for _ in range(num_envs)]
    actions = torch.IntTensor(actions)
    # actions = torch.Tensor(actions, dtype=torch.int).to("cpu")

    print(actions)
    states, rewards, dones, _ = env.step(actions)

    # rollout_size is 1 because we only do 1 step.
    log_probs = torch.rand((1, num_envs))

    mem.add(states, actions, rewards, log_probs, dones)

    assert torch.equal(mem.states[0], torch.FloatTensor(states))
    assert torch.equal(mem.actions, actions)
    assert mem.rewards == torch.FloatTensor(rewards).reshape(-1)
    assert mem.log_probs == log_probs.reshape(-1)
    assert mem.dones == torch.BoolTensor(dones).reshape(-1)
    assert mem.idx == 1
