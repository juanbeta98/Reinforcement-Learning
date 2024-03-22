########## First interactions with GYM library ##########

### Library
import gym

#%% Open AI - Gym Documentation

# Create the environment
env = gym.make('CartPole-v0')

# Number of episodes refering to the number of times the game is played
for i_episode in range(20):
    
    # Process starts when calling reset() method
    # reset() returns an initial observation
    observation = env.reset()
    
    # Number of maximimun time steps on each episode
    for t in range(100):
        
        # render() method displays graphics
        env.render()
        
        # observartion: an enviromental-specific obect representing the state of the environment
        print(observation)
        
        # env.action_space.sample() returns a random action from the space action
        action = env.action_space.sample()
        
        # step(action) method takes as parameters an action from the action_space, it returns 4 values:
        # - observation
        # - reward: amound of reward achieved by the previous actions
        # - done: Boolean indicates if current episode is over and the environment must be reset
        # - info: Disgnositic information usefull for debugging
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

#%% The computer Scientist (CartPole_v1)
### https://www.youtube.com/watch?v=8MC3y7ASoPs
### https://gym.openai.com/envs/CartPole-v1/

### Library

import random

env_name = 'CartPole-v1'
env = gym.make(env_name)

print('Obervation space: ', env.observation_space)
print('Action space: ', env.action_space)

class Agent():
    def __init__(self,env):
        self.action_size = env.action_space.n
        print('Action size: ', self.action_size)
    
    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

agent = Agent(env)
state = env.reset()

for _ in range(200):
    # action = env.action_space.sample()
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()
    
#%% deeplizard (FrozenLake-v0)
### https://www.youtube.com/watch?v=QK_PP_2KgGE
### https://gym.openai.com/envs/FrozenLake-v0/

### Library
import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# Create the environment
name = 'FrozenLake-v0'
env = gym.make(name)

print(env.observation_space)
print(env.action_space)

# Retrive information form the environment
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
# print(q_table)

# Q learning parameters
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# List containing rewards from all episodes
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    
    # Reset environment 
    state = env.reset()
    
    # Variable done wheather or not the episode is finished, initialized as false
    done = False

    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        
        # Exploration-Exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        # Pass the action to the agent
        new_state, reward, done, info = env.step(action)
        
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        
        state = new_state
        rewards_current_episode += reward 
        
        if done == True:
            break
        
    # Exponential rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate * episode)
        
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000 
print('******** Average reward per thousand episodes ********* \n')
for r in rewards_per_thousand_episodes:
    print(count, ': ', str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print('\n \n ******* Q-table **********')
print(q_table)

# Watch the agent interact with the environment with the training
for episode in range(3):
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Print current episode
    print('******** EPISODE', episode + 1, '******** \n \n \n \n')
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        
        # Render environment
        clear_output(wait = True)
        env.render()
        time.sleep(0.3)
        
        # Choose action from experience
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        
        if done:
            clear_output(wait = True)
            env.render()
            if reward == 1:
                print('You reached the goal!! \n \n')
                time.sleep(3)
            else:
                print('You fell through a hole :( \n \n')
                time.sleep(3)
            clear_output(wait = True)
            break
        
        state = new_state

env.close()

#%% Nicholas Rennote (CartPole-v0)
### https://www.youtube.com/watch?v=cO5g5qLrLSo&t=46s
### https://gym.openai.com/envs/CartPole-v1/

# Types of machine learng 
# - Supervised Learning 
# - Unsupervised Learning

# AREA 51
# - Action
# - Reward 
# - Envionment
# - Agent

# Library
import gym
import random

# Create the environment
env = gym.make('CartPole-v0')

# Retrieve information
states = env.observation_space.shape[0]
actions = env.action_space.n
actions

# Taking steps
episodes = 10 
for episode in range(1, episodes + 1):
    
    # Reseting the environment
    state = env.reset()
    done= False
    score = 0
    
    # While the game is still on
    while not done:
        
        # Render the graphical environment
        env.render()
        
        # Taking the corresponding action
        action = random.choice([0,1])
        
        # Updating the state and environment
        n_state, reward, done, info = env.step(action)
        score += reward
    
    print('Episode:{} Score :{}'.format(episode, score))

env.close()


## Deep learning model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Building Deep learning model
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    return model

model = build_model(states, actions)
model.summary()

## Build the agent
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length = 1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy,
                   nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)

    return dqn

# Train the Agent
dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ['mae'])
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

# Retireve information from every episode
scores = dqn.test(env, nb_episodes = 100, visualize = False)
print(np.mean(scores.history['episodes_reward']))

# View agent 
_ = dqn.test(env, nb_episodes = 15, visualize = True)

## Saving and realoading agent
# Save  weights
dqn.save_weights('dqn_weights.h5f', overwrite = True)

# Deleting data
del model
del dqn
del env

# Reconstructing the environment
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr = 1e-3), metrics = ['mae'])

# Re-loading the weights
_ = dqn.load_weights('dqn_weights.h5f')










