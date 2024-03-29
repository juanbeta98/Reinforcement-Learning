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


env = gym.make('CliffWalking-v0', render_mode=None)
epsilon_decaying_value = epsilon / ((EPISODES//1.5) - START_EPSILON_DECAYING)     # Amount of decayment of epsilon    

# Generate the q_table 
q_table = sm.Q_Learning_Agent.generate_q_table('random',env,[env.observation_space.n],low=-2,high=0)

# Rewards
ep_rewards = list()
success = list()
epsilons = list()

#%%

### Training
for episode in range(EPISODES):
    
    episode_reward = 0
    state, info = env.reset()
    done = False
    
    while not done: 

        if np.random.random() > epsilon:                    # Randomize actions with epsilon
            action = np.argmax(q_table[state])              # Action taken from the argmax of the current state
        else:
            action = env.action_space.sample()              # Action taken at random
        
        new_state, reward, terminated, truncated, info = env.step(action)       # Retrieve information
        done = sm.Q_Learning_Agent.evaluate_done(terminated,truncated)

        episode_reward += reward
        
        if not done: 
            q_table = sm.Q_Learning_Agent.update_q_table(q_table,(state,),(new_state,),action,reward,LEARNING_RATE,DISCOUNT)
        
        elif terminated:
            q_table[(state,) + (action, )] = 0        # Update value when goal is reached
        
        state = new_state                 # Update state
    
    epsilon = sm.Q_Learning_Agent.decay_epsilon(epsilon,episode,epsilon_decaying_value,START_EPSILON_DECAYING,EPISODES//1.5)
    
    ep_rewards.append(episode_reward)
    success.append(terminated)
    epsilons.append(epsilon)
     
env.close()
# %%

sum(success)/len(success)

# %%
