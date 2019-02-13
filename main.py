import  random
import os
import gym
import tensorflow as tf
import numpy as np

env = gym.make("MountainCar-v0")
learning_rate= 0.001

max_game_steps = 200
acceptable_score = -198
initial_games = 10000
training_data = []
accepted_scores = []

def create_nn_model(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=input_size, activation='relu'))
    model.add(tf.keras.layers.Dense(58, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
    return model

def create_train_model(training_data):
    inputs = []
    labels = []

    for obs, out in training_data:
        inputs.append(obs)
        labels.append(out)

    inputs = np.array(inputs).reshape(-1, 2)
    labels = np.array(labels).reshape(-1, 3)

    model = create_nn_model(input_size=2, output_size=3)
    model.fit(inputs, labels, epochs=5)

    return model

def prepare_data_with_reward():
    
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        prv_obs = []
        env.reset()

        for step_index in range(max_game_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            if len(prv_obs) > 0:
                game_memory.append([prv_obs, action])

            prv_obs = obs
            position, velocity = obs

            if position > -0.2:
                reward = 1

            score += reward

            if done:
                break
            
        if score >= acceptable_score:
            accepted_scores.append(score)

            for obs, action in game_memory:
                if action == 0:
                    out = [1, 0, 0]
                elif action == 1:
                    out = [0, 1, 0]
                elif action == 2:
                    out = [0, 0, 1]
                
                training_data.append([obs, out])
        

prepare_data_with_reward()


trained_model = create_train_model(training_data)

trained_model.save("model.qnn",overwrite=True)
print("Model saved to model.qnn file.")
print("Simulated games for prepare training data set: " + str(initial_games))
print("Count of training data set: " + str(len(accepted_scores)))
