# rl_lib

## DQN
- Vanilla DQN
- Noisy DQN
- Dualing DQN
- Double DQN
- Prioritiesed Experience Replay DQN
- Rainbow DQN

## Policy Gradient
- A2C (single environment)
- A2C (multiple environments)
- DDPG
- PPP (discrete and continuous)

## Tabular Solutions

- Bellman Equation
- Dynamic Programming
- Q learning

```
m = MultiEnv("CartPole-v0", 8)

next_states = m.reset()
print(next_states)

sleep(1)

next_states, rewards, dones, _  = m.step([1,1,1,1,1,1,1,1])
print(next_states)

sleep(1)

m.close()
```
