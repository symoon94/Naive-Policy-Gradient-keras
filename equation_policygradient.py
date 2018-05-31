#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:07 2018

@author: munsuyeong
"""

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
import random

class PGAgent:
    
    
    def __init__(self, action_size):
        self.load_model = False
        self.input_size = (1) 
        self.action_size = action_size
        #하이퍼파라미터
        self.batch_size = 1
        #모델설정
        self.model = self.build_model()
        self.optimizer = self.optimizer()
        
        #state, action, reward 
        self.states, self.actions, self.rewards = [], [], []
        
        if self.load_model:
            self.model.load_weights("./save_model/equation2.h5")
        
        
    def optimizer(self):
        action = K.placeholder(shape=[None,3])
        reward_value = K.placeholder(shape=[None,])
        policy = self.model.output
        action_prob = K.sum(action * policy, axis=1)
        action_prob = K.clip(action_prob,0.01,0.99)  # log(1)이 되는 것을 방지하기 위해
        cross_entropy = K.log(action_prob)*reward_value
        loss = -K.sum(cross_entropy)
        
        optimizer = Adam(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input,action,reward_value],[loss], updates=updates)
        return train
    
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(15, input_dim=1, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.summary()
        return model
        
    
    def get_action(self, x_data):
        
        if np.random.rand() <= 0.5:
            policy = self.model.predict(x_data)[0]
            policy_buffer = policy
            for i in range(self.action_size):
                if policy[i] >=0.99:
                    for j in range(self.action_size):
                        policy_buffer[j] += 0.3 
                    policy[i] -= 0.9
                    policy_buffer[i] = policy[i]
                    policy = policy_buffer
                    break
            return np.random.choice(self.action_size,1,p=policy)[0]
        
        else:
            return random.randrange(self.action_size)
        
    
    def append_sample(self, state, action, reward):
        if len(self.states) >= self.batch_size:   # memory maxlen 30
            del self.states[0]
            del self.rewards[0]
            del self.actions[0]
            
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
    
    def train_model(self):
        states_buffer = np.zeros((self.batch_size,1))
        reward_buffer = np.zeros((self.batch_size,))
        
        for i in range(self.batch_size):
            reward_buffer[i] = self.rewards[i]
            states_buffer[i] = self.states[i]
            
        self.optimizer([states_buffer,self.actions,reward_buffer])
        
if __name__ == "__main__":
    agent = PGAgent(action_size=3)
    
    delta_x = 0.01
    
    start_values = [0]
    x_data = np.array([start_values])
    
    for episode in range(9000):
        
        x_data = np.clip(x_data, 0., 8.)
        
        action_index = agent.get_action(x_data)
            
        if action_index == 0: # x - 0.01 방향으로 이동
            new_x = x_data - delta_x
                
        elif action_index == 1:  # 정지
            new_x = x_data
                
        elif action_index == 2:  # x + 0.01 방향으로 이동
            new_x = x_data + delta_x
            
        pre_reward = 3*np.square(x_data-6)*(x_data-2) 
        post_reward = 3*np.square(new_x-6)*(new_x-2) 
        reward = post_reward - pre_reward
             
        agent.append_sample(x_data, action_index, reward)
        
        
        if episode >= agent.batch_size :
            agent.train_model()
            
        x_data = new_x
        
        
        print( 'episode:',episode, " action_index:",action_index
              ," final_value:", x_data, " delta_reward:",reward)
        
        if episode % 10000 == 0:
            agent.model.save_weights("./save_model/equation2.h5")
        

print('===================================')
print( '아래는 각 x값에서의 action probability' )
for i in range (1,10):
    print('if x =' ,i ,'? ',agent.model.predict(np.array([[i]])) )   
