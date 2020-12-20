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

def test_calculating_discounted_returns_with_done_and_last_value():
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
  mem.calculate_discounted_returns([0.5,0.5], [0,0])

  # Check discounted returns
  dr = [0.000000,1.427058,1.441472,1.456033,1.470740,0.475495,0.480298,0.485150,0.490050,0.495000]
  discounted_returns = torch.FloatTensor([dr, dr])
  discounted_returns = torch.transpose(discounted_returns, 1, 0)
  # import pdb; pdb.set_trace()
  assert torch.allclose(mem.discounted_returns, discounted_returns)


def test_calculating_discounted_returns_with_done_and_neg_last_value():
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
  mem.calculate_discounted_returns([-0.5,-0.5], [0,0])

  # Check discounted returns
  dr = [0.00000000,0.51354039,0.51872766,0.52396733,0.52925992,-0.47549507,-0.48029804,-0.48514953,-0.49005002,-0.49500000]
  discounted_returns = torch.FloatTensor([dr, dr])
  discounted_returns = torch.transpose(discounted_returns, 1, 0)
  assert torch.allclose(mem.discounted_returns, discounted_returns)

def test_sample_memory():
  rollout_size = 3
  num_envs = 2
  mem, env = setup(rollout_size, num_envs)

  mem.rewards = torch.FloatTensor([[0, 0], [0, 0], [1, 1]])
  mem.states = torch.zeros((rollout_size, num_envs) + env.observation_space.shape)
  mem.actions = torch.zeros((rollout_size, num_envs) + env.action_space.shape)
  mem.log_probs = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
  mem.dones = torch.FloatTensor([[0, 0], [0, 0], [0, 0]])
  
  mem.calculate_discounted_returns([0,0], [0,0])
  s, a, lp, dr = mem.sample([0,2,4])
  
  assert torch.equal(lp, torch.FloatTensor([1,3,5]))
  assert torch.allclose(dr, torch.FloatTensor([[0.9801,0.9900, 1]]))

def test_getting_mini_batch_eql_number():
  mem, env = setup(rollout_size=100, num_envs=2)
  mini_batch_idxs = mem.get_mini_batch_idxs(10)

def test_getting_mini_batch_eql_number():
  mem, env = setup(rollout_size=100, num_envs=2)
  mini_batch_idxs = mem.get_mini_batch_idxs(9)

  assert len(mini_batch_idxs) == 23
  assert len(mini_batch_idxs[22]) == 2
