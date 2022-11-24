import gym
import torch
import matplotlib


from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("MsPacman-v0")

model = A2C("MlpPolicy", env, n_steps=10, verbose=1)
model.learn(total_timesteps=100, log_interval=1)
model.save("a2c_MsPacman")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_MsPacman")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs =env.reset()