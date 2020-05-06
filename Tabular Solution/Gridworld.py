import numpy as np
from copy import deepcopy

DEFAULT = [
  [0  ,0   ,0  ,0  ,1  ],
  [0  ,'x' ,'x',0  ,-1 ],
  [0  ,0   ,0  ,0  ,0  ],
]

class Gridworld:
    def __init__(self, architecture, walking_penality=-0.1):
        self.grid = deepcopy(architecture)
        self.available_states = []
        self.terminal_states = []
        self.available_moves = {}
        
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.move_prob = 0.8

        for y in range(self.height):
            for x in range(self.width):

                # Check if state is a valid position to be in
                if (self.grid[y][x] != 'x'):

                    # Check if state is a terminal position
                    if (self.grid[y][x] > 0 or self.grid[y][x] < 0):
                        self.terminal_states.append((y,x))
                        
                    if (self.grid[y][x] == 0):
                        self.available_states.append((y,x))
                        # set move penalty
                        self.grid[y][x] = -0.1
                
                    
                    
        for y, x in self.available_states:
            self.available_moves[(y, x)] = self.get_valid_moves((y, x))

    def get_valid_moves(self, state):
        y = state[0]
        x = state[1]
        valid_moves = []
        
        # Action Up
        new_loc = (y - 1 if y - 1 > 0 else 0, x)
        if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
            valid_moves.append(("UP", self.grid[new_loc[0]][new_loc[1]]))
                               
        # Action Right
        new_loc = (y, x + 1 if x + 1 < self.width else x)
        if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
            valid_moves.append(("RIGHT", self.grid[new_loc[0]][new_loc[1]]))

        # Action Down
        new_loc = (y + 1 if y + 1 < self.height else y, x)
        if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
            valid_moves.append(("DOWN", self.grid[new_loc[0]][new_loc[1]]))

                            
        # Action Left
        new_loc = (y, x - 1 if x - 1 > 0 else 0)
        if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
            valid_moves.append(("LEFT", self.grid[new_loc[0]][new_loc[1]]))
        
        return valid_moves

    def reward_of_action(self, action, state):
        y = state[0]
        x = state[1]
        
        # returns (s', R(s'))
        if action == "UP":
            new_loc = (y - 1 if y - 1 > 0 else 0, x)
            if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
                return new_loc , self.grid[new_loc[0]][new_loc[1]]
            else:
                return (y, x) , self.grid[y][x]

        elif action == "RIGHT":
            new_loc = (y, x + 1 if x + 1 < self.width else x)
            if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
                return new_loc , self.grid[new_loc[0]][new_loc[1]]
            else:
                return (y, x) , self.grid[y][x]
        

        elif action == "DOWN":
            new_loc = (y + 1 if y + 1 < self.height else y, x)
            if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
                return new_loc , self.grid[new_loc[0]][new_loc[1]]
            else:
                return (y, x) , self.grid[y][x]

        elif action == "LEFT":
            new_loc = (y, x - 1 if x - 1 > 0 else 0)
            if (self.grid[new_loc[0]][new_loc[1]] != 'x'):
                return new_loc , self.grid[new_loc[0]][new_loc[1]]
            else:
                return (y, x) , self.grid[y][x]
    
    def transition_probabilities(self, action, state):
        y = state[0]
        x = state[1]
        # returns list of (probability, reward, next_state)
        probs = []
    
        next_state, next_reward = self.reward_of_action(action, (y, x))
        probs.append((self.move_prob, next_reward, next_state))
        
        
        disobey_probs = 1 - self.move_prob
        
        
        if action == "UP" or action == "DOWN":
            next_state, next_reward = self.reward_of_action("LEFT", (y, x))
            probs.append((disobey_probs / 2, next_reward, next_state))
            next_state, next_reward = self.reward_of_action("RIGHT", (y, x))
            probs.append((disobey_probs / 2, next_reward, next_state))
        
        if action == "LEFT" or action == "RIGHT":
            next_state, next_reward = self.reward_of_action("UP", (y, x))
            probs.append((disobey_probs / 2, next_reward, next_state))
            
            next_state, next_reward = self.reward_of_action("DOWN", (y, x))
            probs.append((disobey_probs / 2, next_reward, next_state))
        
        return probs

