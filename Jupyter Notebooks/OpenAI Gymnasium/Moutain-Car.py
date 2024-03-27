#%%



# %%
import numpy as np
import gymnasium as gym
import support_modules as sm

# Parameters
LEARNING_RATE = 0.1         # How fast does the agent learn
DISCOUNT = 0.95             # How important are future actions

EPISODES = 2000            # Number of episodes
SHOW_EVERY = 250           # Parameter to show progress

epsilon = 0.5                           # Rate at which random actions will be 
START_EPSILON_DECAYING = 1              # First episode at which decay epsilon
END_EPSILON_DECAYING = 1250             # Last episode at which decay epsilon


env = gym.make('MountainCar-v0',render_mode=None)

# Generate the discrete state space and the interval of each discrete space 
discrete_state_space,discrete_state_intervals = sm.generate_discrete_states(20,env)

# Generate the q_table 
q_table = sm.generate_q_table('random',env,discrete_state_space,low=-2,high=0)

# Rewards
ep_rewards = []


### Training
for episode in range(EPISODES):
    
    episode_reward = 0
    state, info = env.reset()
    discrete_state = sm.get_discrete_state(state,env,discrete_state_intervals)        # Discrete initial state
    done = False
    
    while not done: 

        if np.random.random() > epsilon:                    # Randomize actions with epsilon
            action = np.argmax(q_table[discrete_state])     # Action taken from the argmax of the current state
        else:
            action = env.action_space.sample()              # Action taken ramdomly
        
        new_state, reward, terminated, truncated, info = env.step(action)       # Retrieve information
        done = sm.define_done(terminated,truncated)

        episode_reward += reward
        
        new_discrete_state = sm.get_discrete_state(new_state,env,discrete_state_intervals)  # Discretize new state
        
        if not done: 
            q_table = sm.update_q_table(q_table,discrete_state,new_discrete_state,action,LEARNING_RATE,DISCOUNT)
        
        elif terminated:
            print(f'We made it in episode {episode}')
            q_table[discrete_state + (action, )] = 0        # Update value when goal is reached
        
        discrete_state = new_discrete_state                 # Update state
    
    epsilon = sm.decay_epsilon(epsilon,episode,START_EPSILON_DECAYING,END_EPSILON_DECAYING)
    
    ep_rewards.append(episode_reward)
    
        
env.close()


# %%



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_moving_average(binary_numbers, window_size):
    # Convert binary numbers to pandas Series
    series = pd.Series(binary_numbers)
    
    # Compute the moving average
    moving_avg = series.rolling(window=window_size).mean()
    
    # Plot the binary numbers and the moving average
    plt.figure(figsize=(10, 6))
    # plt.plot(binary_numbers, label='Binary Numbers', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(moving_avg, label=f'Moving Average (Window Size: {window_size})', color='purple', linewidth=2)
    
    # Add labels and legend
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Moving Average of Binary Numbers', fontsize=14)
    plt.legend(fontsize=10)
    
    # Customize grid lines
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
binary_numbers = [np.random.choice([0,1]) for i in range(100)]  # Example binary numbers
window_size = 5  # Example window size for the moving average
plot_moving_average(binary_numbers, window_size)


# %%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_moving_average(series_list, window_size):
    # Create a DataFrame from the list of series
    df = pd.DataFrame(series_list).T
    
    # Compute the moving average for each series
    moving_avg = df.rolling(window=window_size).mean()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(moving_avg.columns)))

    # Plot the original series and the moving average for each
    plt.figure(figsize=(10, 6))
    for i, series in enumerate(moving_avg.columns):
        plt.plot(moving_avg[series], label=f'Moving Average {i + 1}', linewidth=2, color=colors[i])

    # Add labels and legend
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Moving Average of Multiple Series', fontsize=14)
    plt.legend(fontsize=10)
    
    # Customize grid lines
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
series_list = [
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
]  # Example series list
window_size = 3  # Example window size for the moving average
plot_moving_average(series_list, window_size)


# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def compute_moving_average(series_list, window_size):
    # Create a DataFrame from the list of series
    df = pd.DataFrame(series_list).T
    
    # Compute the moving average for each series
    moving_avg = df.rolling(window=window_size).mean()
    
    return moving_avg

# Example usage
series_list = [
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
]  # Example series list
window_size = 3  # Example window size for the moving average
moving_avg = compute_moving_average(series_list, window_size)
print(moving_avg)


def plot_moving_average(series_list, window_size):
    # Compute moving averages for each series
    moving_avg = compute_moving_average(series_list, window_size)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for i, col in enumerate(moving_avg.columns):
        plt.plot(moving_avg.index, moving_avg[col], label=f'Series {i+1} Moving Average')

    # Add title and labels
    plt.title(f'Moving Average with Window Size {window_size}')
    plt.xlabel('Time')
    plt.ylabel('Moving Average')

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
series_list = [
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
]  # Example series list
window_size = 3  # Example window size for the moving average

plot_moving_average(series_list, window_size)

# %%
