#%%
import gymnasium as gym
env = gym.make('CartPole-v1',render_mode=None)

# %%
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)           # Discretizing the state space
# %%
