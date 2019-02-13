import gym
import time
import random
import tensorflow as tf
import numpy as np

env = gym.make("MountainCar-v0")
model = tf.keras.models.load_model("model.qnn", compile=True)
game_amount = 100
goal_steps = 200

for each_game in range(game_amount):
    prev_obs = []
    env.reset()
    for step_index in range(goal_steps):
        env.render()
        if step_index == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, 2)))
        
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        if done:
            break

        time.sleep(0.05)
