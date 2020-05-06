import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        
        self.noise = OUNoise(action_space)
        
        self.head = nn.Sequential(
            nn.Linear(state_space, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.head(x)
    
    def act(self, state, add_noise=True):
        
        state = torch.from_numpy(state).float().to(device)
        
        action = self.forward(state).cpu().data.numpy()
        if add_noise:
            action += self.noise.noise()

        return np.clip(action, -1, 1)

class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(state_space, 1024),
            nn.ReLU(),
        )
        
        self.body = nn.Sequential(
            nn.Linear(1024 + action_space, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
    
    def forward(self, x, actions):
        x = self.head(x)
        x = self.body(torch.cat((x, actions), dim=1))
        return x