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

def test_add_step_to_memory():
    rollout_size = 10
    num_envs = 2
    mem, env = setup(rollout_size, num_envs)
    env.reset()

    # Take Dummy Step
    actions = [env.action_space.sample() for _ in range(num_envs)]
    actions = torch.IntTensor(actions)
    states, rewards, dones, _ = env.step(actions)
    
    # Only 1 because single step taken
    log_probs = torch.rand((1, num_envs))
    
    # Add to memory
    mem.add(states, actions, rewards, log_probs, dones)

    assert torch.equal(mem.states[0], torch.FloatTensor(states))
    assert torch.equal(mem.actions[0], actions.type(torch.FloatTensor))
    assert torch.equal(mem.rewards[0], torch.FloatTensor(rewards).reshape(-1))
    assert torch.equal(mem.log_probs[0], log_probs.type(torch.FloatTensor).reshape(-1))
    assert torch.equal(mem.dones[0], torch.FloatTensor(dones).reshape(-1))
    assert mem.idx == 1

def test_calculating_discounted_returns_single_reward():
    mem, env = setup(rollout_size=10, num_envs=2)

    # set up rewards & dones
    r = [0,0,0,0,1,0,0,0,0,0]
    rewards = torch.FloatTensor([r, r])
    rewards = torch.transpose(rewards, 1, 0)

    d = [0,0,0,0,0,0,0,0,0,0]
    dones = torch.FloatTensor([d, d])
    dones = torch.transpose(dones, 1, 0)

    mem.rewards = rewards
    mem.dones = dones

    # Calculate returns
    mem.calculate_discounted_returns([0,0], [0,0])

    # Check discounted returns
    dr = [0.9606,0.9703,0.9801,0.9900,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000]
    discounted_returns = torch.FloatTensor([dr, dr])
    discounted_returns = torch.transpose(discounted_returns, 1, 0) 
    assert torch.allclose(mem.discounted_returns, discounted_returns)

def test_calculating_discounted_returns_with_done():
    mem, env = setup(rollout_size=10, num_envs=2)

    # set up rewards & dones
    r = [0,0,0,0,1,0,0,0,0,0]
    rewards = torch.FloatTensor([r, r])
    rewards = torch.transpose(rewards, 1, 0)

    d = [0,1,0,0,0,0,0,0,0,0]
    dones = torch.FloatTensor([d, d])
    dones = torch.transpose(dones, 1, 0)

    mem.rewards = rewards
    mem.dones = dones

    # Calculate returns
    mem.calculate_discounted_returns([0,0], [0,0])

    # Check discounted returns
    dr = [0.0000,0.9703,0.9801,0.9900,1.0000,0.0000,0.0000,0.0000,0.0000,0.0000]
    discounted_returns = torch.FloatTensor([dr, dr])
    discounted_returns = torch.transpose(discounted_returns, 1, 0) 
    assert torch.allclose(mem.discounted_returns, discounted_returns)
