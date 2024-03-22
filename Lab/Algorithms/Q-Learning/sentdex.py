### sentdex 
### https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=3

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)


##### Q-Learning
# Parameters
LEARNING_RATE = 0.1         # How fast does the agent learn
DISCOUNT = 0.95             # How important are future actions

EPISODES = 2000            # Number of episodes
SHOW_EVERY = 250           # Parameter to show progress

epsilon = 0.5                           # Rate at which random actions will be 
START_EPSILON_DECAYING = 1              # First episode at which decay epsilon
END_EPSILON_DECAYING = EPISODES // 2    # Last episode at which decay epsilon
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)     # Amount of decayment of epsilon                                                                                 # per episode
# Parameters

# Discretizing states
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)           # Discretizing the state space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) /\
    DISCRETE_OS_SIZE                                                # Range per slot 


# Q-table has 3 dimensions, two that conform the action space and one for the actions.
# Therefore, in every space the q value for that position, velocity and action will be stored
q_table = np.random.uniform(low = -2, high = 0, size = 
                            (DISCRETE_OS_SIZE + [env.action_space.n]))   # Initialize q table with random values

def get_discrete_state(state):                          # Helper function that discretizes a given state
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
# Discretizing states

# Rewards
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg':[], 'min': [], 'max': []}
# Rewards
### Q-Learning


### Iterative process
for episode in range(EPISODES):
    
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0:                           # Check condition for rendering and printing the episode
        print(episode)
        render = True
    else:
        render = False
        
    discrete_state = get_discrete_state(env.reset())        # Discrete initial state
    done = False
    
    while not done: 
        
        if np.random.random() > epsilon:                    # Randomize actions with epsilon
            action = np.argmax(q_table[discrete_state])     # Action taken from the argmax of the current state
        else:
            action = env.action_space.sample()              # Action taken ramdomly
        
        new_state, reward, done, _ = env.step(action)       # Retrieve information
        episode_reward += reward
        
        new_discrete_state = get_discrete_state(new_state)  # Discretize new state
        
        if render:
            env.render()
        
        if not done: 
            
            max_future_q = np.max(q_table[new_discrete_state])          # Maximum value of arriving state
            current_q = q_table[discrete_state + (action, )]            # Value of current state and action
            
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
                (reward + DISCOUNT * max_future_q)                      # Q-Learning formula
            
            q_table[discrete_state + (action,)] = new_q     # Update Q Value for current state and action
        
        elif new_state[0] >= env.goal_position:
            print(f'We made it in episode {episode}')
            q_table[discrete_state + (action, )] = 0        # Update value when goal is reached
        
        discrete_state = new_discrete_state                 # Update state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:       # Decay epsilon
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()
 


