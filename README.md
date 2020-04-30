### Multi Env example

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