import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd


class Q_Learning_Agent():
    @staticmethod
    def generate_discrete_states(partitions:int, env:gym.Env) -> tuple:
        """
        Generate discrete states based on the given partition sizes.

        Parameters:
        - partitions (int): The number of partitions for each dimension of the state space.
        - env: The environment with the observation space.

        Returns:
        - state_spaces (list): A list containing the number of discrete states for each dimension.
        - state_intervals (ndarray): An array containing the size of each state interval for each dimension.
        """
        # Calculate the number of discrete states for each dimension
        state_spaces = [partitions] * len(env.observation_space.high)
        
        # Calculate the size of each state interval for each dimension
        state_intervals = (env.observation_space.high - env.observation_space.low) / state_spaces

        return state_spaces, state_intervals


    @staticmethod
    def generate_q_table(init_strategy:str, env:gym.Env, discrete_state_space:list, **kwargs) -> np.ndarray:
        """
        Generate a Q-table for the given environment and initialization strategy.

        Parameters:
        - init_strategy (str): The initialization strategy for the Q-table. Supported strategies are: 'random', 'zeros', and 'init_value'.
        - env (gym.Env): The Gym environment.
        - discrete_state_space (int): The number of discrete states in the state space.
        - **kwargs: Additional keyword arguments depending on the initialization strategy.
            - For 'random': 'low' (float) and 'high' (float) specify the range of random values.
            - For 'init_value': 'value' (float) specifies the initial value for all Q-table entries.

        Returns:
        - q_table (np.ndarray): The initialized Q-table.
        """
        # Check the initialization strategy and create the Q-table accordingly
        if init_strategy == 'random':
            q_table = np.random.uniform(low=kwargs['low'], high=kwargs['high'], size=(discrete_state_space + [env.action_space.n]))
        elif init_strategy == 'zeros':
            q_table = np.zeros(shape=(discrete_state_space, env.action_space.n))
        elif init_strategy == 'init_value':
            q_table = np.full(shape=(discrete_state_space, env.action_space.n), fill_value=kwargs['value'])
        else:
            raise ValueError(f"Invalid initialization strategy: {init_strategy}")

        return q_table


    @staticmethod
    def get_discrete_state(state, env:gym.Env, state_intervals) -> tuple:
        """
        Convert a continuous state into a discrete state representation.

        Parameters:
        - state (np.ndarray): The continuous state vector.
        - env (gym.Env): The Gym environment.
        - state_intervals (np.ndarray): The intervals defining the discrete state space.

        Returns:
        - discrete_state (tuple): The discrete state representation.
        """
        # Calculate the discrete state using intervals
        discrete_state = (state - env.observation_space.low) / state_intervals
        
        # Convert to integer and return as a tuple
        return tuple(discrete_state.astype(int))


    @staticmethod
    def evaluate_done(terminated:bool, truncated:bool) -> bool:
        """
        Define the termination condition for an episode based on termination and truncation flags.

        Parameters:
        - terminated (bool): Flag indicating whether the episode is terminated.
        - truncated (bool): Flag indicating whether the episode is truncated.

        Returns:
        - done (bool): True if the episode is done, False otherwise.
        """
        # If either terminated or truncated is True, the episode is done
        if terminated or truncated:
            return True
        else:
            return False


    @staticmethod
    def decay_epsilon(epsilon: float, episode: int, epsilon_decaying_value:float, START_EPSILON_DECAYING: int, END_EPSILON_DECAYING: int) -> float:
        """
        Decay the exploration rate epsilon linearly over a specified range of episodes.

        Parameters:
        - epsilon (float): The current value of the exploration rate.
        - episode (int): The current episode number.
        - START_EPSILON_DECAYING (int): The episode at which epsilon decay starts.
        - END_EPSILON_DECAYING (int): The episode at which epsilon decay ends.

        Returns:
        - epsilon (float): The updated value of the exploration rate after decay.
        """
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            # Linearly decay epsilon over the specified range of episodes
            epsilon -= epsilon_decaying_value
        return epsilon


    @staticmethod
    def update_q_table(q_table: np.ndarray, discrete_state: tuple, new_discrete_state: tuple, action: int,
                    reward: float, LEARNING_RATE: float, DISCOUNT: float) -> np.ndarray:
        """
        Update the Q-table based on the observed reward and the transition to a new state.

        Parameters:
        - q_table (np.ndarray): The Q-table containing Q-values for state-action pairs.
        - discrete_state (tuple): The discrete representation of the current state.
        - new_discrete_state (tuple): The discrete representation of the new state.
        - action (int): The action taken in the current state.
        - reward (float): The reward received after taking the action.
        - LEARNING_RATE (float): The learning rate controlling the impact of new information.
        - DISCOUNT (float): The discount factor for future rewards.

        Returns:
        - q_table (np.ndarray): The updated Q-table.
        """
        max_future_q = np.max(q_table[new_discrete_state])  # Maximum Q-value for the new state
        current_q = q_table[discrete_state + (action,)]     # Q-value for the current state-action pair

        # Update the Q-value for the current state-action pair using the Q-learning formula
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update the Q-value in the Q-table
        q_table[discrete_state + (action,)] = new_q

        return q_table


class visualizations():
    @staticmethod
    def plot_series(series_list, labels, **kwargs):
        colors = plt.cm.viridis(np.linspace(0, 1, len(series_list)))

        if labels == ['epsilon']:
            colors = ['gold']
            
        # Plot the original series and the moving average for each
        plt.figure(figsize=(10, 6))
        for i, series_list in enumerate(series_list):
            plt.plot(series_list, label=f'{labels[i]}', linewidth=2, color=colors[i])

        # Add labels and legend
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel(kwargs['ylabel'], fontsize=12)
        plt.title(kwargs['title'], fontsize=14)
        plt.legend(fontsize=10)
        
        # Customize grid lines
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Show the plot
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_moving_average(series_list, labels, window_size, **kwargs):
        # Create a DataFrame from the list of series
        df = pd.DataFrame(series_list).T
        
        # Compute the moving average for each series
        moving_avg = df.rolling(window=window_size).mean()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(moving_avg.columns)))

        # Plot the original series and the moving average for each
        plt.figure(figsize=(10, 6))
        for i, series in enumerate(moving_avg.columns):
            plt.plot(moving_avg[series], label=f'{labels[i]}', linewidth=2, color=colors[i])

        # Add labels and legend
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel(kwargs['ylabel'], fontsize=12)
        plt.title(kwargs['title'], fontsize=14)
        plt.legend(fontsize=10)
        
        # Customize grid lines
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Show the plot
        plt.tight_layout()
        plt.show()