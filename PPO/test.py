from Config import Config
from procgen_env import make_procgen_env
from procgen import ProcgenEnv

env = make_procgen_env("starpilot", 8, "cpu")
print(env.observation_space)
