#%%
import gymnasium as gym
import support_modules as sm
import numpy as np
import ipywidgets as widgets

# Fixed Parameters
LEARNING_RATE = 0.1         # How fast does the agent learn
DISCOUNT = 0.95             # How important are future actions

EPISODES =2000            # Number of episodes

epsilon = 0.5                           # Exploration rate
START_EPSILON_DECAYING = 1              # First episode at which decay epsilon
END_EPSILON_DECAYING = 1250             # Last episode at which decay epsilon



# Rewards
ep_rewards = list()
success = list()
epsilons = list()

#%%
env = gym.make('Pendulum-v1', render_mode=None)

discrete_partitions = 30

# Discrete actions
discrete_actions = [-2,-1.5,-1,-0.75,-0.5,-0.25,-0.1,0]
discrete_actions.extend([-i for i in discrete_actions[::-1]])
discrete_actions.remove(0)

epsilon_decaying_value = epsilon / ((EPISODES//1.5) - START_EPSILON_DECAYING)     # Amount of decayment of epsilon    

# Generate the discrete state space and the interval of each discrete space 
discrete_state_space,discrete_state_intervals = sm.Q_Learning_Agent.generate_discrete_states(discrete_partitions,env)

# Generate the q_table 
q_table = sm.Q_Learning_Agent.generate_q_table('random',len(discrete_actions),discrete_state_space,low=-2,high=0)

# Rewards
ep_rewards = list()
success = list()
epsilons = list()


### Training
for episode in range(EPISODES):
    
    episode_reward = 0
    state, info = env.reset()
    discrete_state = sm.Q_Learning_Agent.get_discrete_state(state,env,discrete_state_intervals,discrete_partitions)        # Discrete initial state
    done = False
    
    while not done: 

        if np.random.random() > epsilon:                    # Randomize actions with epsilon
            action = np.argmax(q_table[discrete_state])     # Action taken from the argmax of the current state
        else:
            action = np.random.randint(low=0, high=len(discrete_actions))              # Action taken at random
        
        real_action = discrete_actions[action]
        
        new_state, reward, terminated, truncated, info = env.step((real_action,))       # Retrieve information
        done = sm.evaluate_done(terminated,truncated)

        episode_reward += reward
        
        new_discrete_state = sm.Q_Learning_Agent.get_discrete_state(new_state,env,discrete_state_intervals,discrete_partitions)  # Discretize new state
        
        if not done: 
            q_table = sm.Q_Learning_Agent.update_q_table(q_table,discrete_state,new_discrete_state,action,reward,LEARNING_RATE,DISCOUNT)
        
        # elif terminated:
        #     q_table[discrete_state + (action, )] = 0        # Update value when goal is reached
        
        discrete_state = new_discrete_state                 # Update state
    
    epsilon = sm.Q_Learning_Agent.decay_epsilon(epsilon,episode,epsilon_decaying_value,START_EPSILON_DECAYING,EPISODES//1.5)
    
    ep_rewards.append(episode_reward)
    success.append(terminated)
    epsilons.append(epsilon)
     