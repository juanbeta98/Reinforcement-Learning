### https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html

#%% Introduction
import gym
from stable_baselines3 import A2C

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()
      

#%% One line implelentation
      
from stable_baselines3 import A2C

model = A2C('MlpPolicy', 'CartPole-v1').learn(10000)   

#%% Check if TPPenv is a Gym 

from stable_baselines3.common.env_checker import check_env
from TPPscript import TPPenv

env = TPPenv()
check_env(env)



