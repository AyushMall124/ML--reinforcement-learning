import random
import gym
import trial_lkk
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import math
import logging
import pandas as pd
import matplotlib.pyplot as plt
logging.getLogger('tensorflow').setLevel(logging.ERROR)

trial_range = 500

def huber_loss( y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


def init(  ):
    model = Sequential()
    model.add(Dense(32, input_dim=states, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.compile(
                loss=huber_loss,
                optimizer=Adam(lr=learn_rate))
    return model

def predict_action( state ):
    act_values = model.predict(state)
    return np.argmax(act_values[0])  


env = gym.make('CartPole-v1')

states = env.observation_space.shape[0]
actions = env.action_space.n

learn_rate = 0.001
model = init()
model.load_weights("Models/cartpole-lkk-task3.h5")

mem = deque(maxlen=2000)
gam = 0.95    

done = False


scores = pd.Series()
for t in range(trial_range):
    state = env.reset()
    state = np.reshape(state, [1, states])
    for time in range(500):
        env.render(mode='human')
        action = predict_action(state)
        next_state, reward, done, _ = env.step(action)
        x,cart_vel,theta,pole_vel = next_state

        next_state = np.reshape(next_state, [1, states])
        state = next_state

        if done or time+1 == 500:
            scores = scores.append( pd.Series(time+1) )
            moving_avg = scores.rolling(20,min_periods=1).mean().iloc[-1]
            print("trial:",t,"/",trial_range,"steps:",time+1,"  moving_avg: %0.2f"%moving_avg)
            break
    


            

