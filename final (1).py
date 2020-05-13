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
    if np.random.rand() <= eps:
        return random.randrange(actions)
    act_values = model.predict(state)
    return np.argmax(act_values[0])  

def retrain( batch ):
    global eps
    minibatch = random.sample(mem, batch)
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(state)
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + gam * np.amax( target_model.predict(next_state)[0] )
        model.fit(state, target, epochs=1, verbose=0)
    if eps > eps_min:
        eps *= eps_decay



env = gym.make('lkk-v0')

states = env.observation_space.shape[0]
actions = env.action_space.n

learn_rate = 0.001
model = init()
model.load_weights("./save/finalmodel3.h5")
# target = model

target_model = init()
target_model.set_weights(model.get_weights())

mem = deque(maxlen=2000)
gam = 0.95    
eps = 0.01
eps_min = 0.01
eps_decay = 0.99

done = False
batch = 30

final = pd.DataFrame(columns=['scores'],index=range(trial_range))

scores = pd.Series()
for t in range(trial_range):
    state = env.reset()
    state = np.reshape(state, [1, states])
    for time in range(500):
        # env.render(mode='human')
        action = predict_action(state)
        next_state, reward, done, _ = env.step(action)
        x,cart_vel,theta,pole_vel = next_state

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.6
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.8
        reward = r1 + r2

        next_state = np.reshape(next_state, [1, states])
        mem.append((state, action, reward, next_state, done))
        state = next_state

        if done or time+1 == 500 :
            final.loc[t,'scores'] = time+1
            target_model.set_weights(model.get_weights())
            scores = scores.append( pd.Series(time+1) )
            moving_avg = scores.rolling(20,min_periods=1).mean().iloc[-1]
            print("trial:",t,"/",trial_range,"steps:",time+1,"  moving_avg: %0.2f"%moving_avg,"epsilon: %0.2f"%eps)
            break
        if len(mem) > batch:
            retrain( batch )
    if t % 10 == 0:
        model.save_weights("./save/finalmodel3-3.h5")
    final.to_csv('final2.csv')
            

