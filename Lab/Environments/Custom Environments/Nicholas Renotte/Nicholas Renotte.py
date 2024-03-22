### https://www.youtube.com/watch?v=bD6V3rcr_54&t=3s

######## Creating the model ########
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

# Initializing the shower class
# Inherit from the class Env
class ShowEnv(Env):
    
    # Initializing method
    def __init__(self):
        
        # Actions that can be taken
        ## Discrete: A discrete number of values 
        self.action_space = Discrete(3)
        
        # Temperature array
        ## Box: Allows more flexibility on its containment. A continuous value, n-dimensiona, arrays, etc.
        self.observation_space = Box(low = np.array([0]), high = np.array([100]))
        
        # Set start temp
        self.state = 38 + random.randint(-3, 3)
        
        # Set shower length
        self.shower_length = 60
    
    # Step method every time a step takes place
    def step(self, action):
        
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 =  0
        # 2 -1 = +1 temperature
        self.state += action - 1
        
        # Reduce shower length by a second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
        
        # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False
        
        # Apply temperature noise
        self.state += random.randint(-1, 1) 
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info
    
    # Render if the instance wants to be visualized
    def render(self):
        
        # Implement visualization
        pass
    
    # Reset the environment after an episode is over
    def reset(self):
        
        # Reset shower temperature
        self.state = 38 + random.randint(-3, 3)
        
        # Reset shower time 
        self.shower_length = 60
        return self.state
    
env = ShowEnv()

env.action_space.sample()
env.observation_space.sample()

# episodes = 10
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         # env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward 
    
#     print(f'Episode {episode}   Score :{score}')

######## Creating the Deep Learnign model with Keras ########   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
actions = env.action_space.n

def build_model(states, actions):
    
    model = Sequential()
    model.add(Dense(24, activation = 'relu', input_shape = states))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    
    return model 

model = build_model(states, actions)
model.summary()

######## Build Agent with Keras RL ########   
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length = 1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions,
                   nb_steps_warmup = 10, target_model_update = 1e-2)
    
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ['mae'])
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)