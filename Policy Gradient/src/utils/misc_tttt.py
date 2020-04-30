def untrained_simulation():
  agent = Agent(state_size=env.stateSize, action_size=env.actionSize, seed=0)

  # watch an untrained agent
  state = env.reset()
  
  for j in range(1000):
    action = env.randomAction()
    state, reward, done, _ = env.step(action)
    if np.any(done):
      break 
        
  env.display()

def watch_trained_agent():
  # load the weights from file
  agent = Agent(state_size=8, action_size=4, seed=0)
  agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

  for i in range(3):
      state = env.reset()
      env.render(mode='rgb_array')
      for j in range(200):
          action = agent.act(state)
          img.set_data(env.render(mode='rgb_array')) 
          plt.axis('off')
          display.display(plt.gcf())
          display.clear_output(wait=True)
          state, reward, done, _ = env.step(action)
          r += reward
          if done:
              break 
              
  env.close()