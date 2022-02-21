'''
=====================================================================
Reinforcement Learning Framwork File
This file contains reinforcement learning framworks, using Keras
    Q-Leaning algorithm
    Double Q-Learning algorithm
    Actor-Critic algorithm
    Advantange Actor-Critic(A2C) algorithm - (not finish)
    Deep Deterministic Policy Gradient (DDPG)
    Multi-agent Deep Deterministic Policy (MADDPG)
Adjustable parameter are under '研究室'

強化学習フレームワークファイル
このファイルには、Kerasを使用した強化学習フレームワークが含まれています
     Q学習アルゴリズム
     二重Q学習アルゴリズム
     Actor-Criticアルゴリズム
     Advantange Actor-Critic（A2C）アルゴリズム-（未完成）
     ディープデターミニスティックポリシーグラディエント（DDPG）
     マルチエージェントディープデターミニスティックポリシー（MADDPG）
調整可能なパラメータは「研究室」の下にあります
=====================================================================
'''

'''
# Readme: What have been fixed
    * Q-learning
    * Double Q-learning


'''
import math
import os
import datetime
import random
from collections import deque

import numpy as  np
import tensorflow as tf
random.seed(3407)
np.random.seed(3407)

tensor_seed = 3407
tf.random.set_seed(tensor_seed)


import tensorflow.keras.backend as K
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
#--------------------------------
# DDPG_Actor_Critic class with keras
#--------------------------------
# Ornstein-Uhlenbeck noise
class OUNoise():
    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = 0.0001
    def gen_noise(self,x):
        return self.theta*(self.mu-x)*self.dt + self.sigma*np.random.randn(1)


class DDPG_LSTM_head:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_ob,num_action,mu,theta,sigma,window):
        self.number = 1
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin= 0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.temprp = deque(maxlen = max_mem)
        self.num_state = num_ob # as list

        self.num_action = num_action
        self.batch_size = 32
        self.tau = 0.05 # soft update
        self.var_actor = None
        self.var_critic= None
        self.noise = []#[NoiseofAction1,NoiseofAction2,...]
        self.update_num =0
        self.update_lr = 0
        self.c_loss = []
        self.window= window

        self.create_noise(mu,theta,sigma)
        # Actor Model
        '''
        During training
          Actor           : Trained
          Actor_target    : Not Trained, but updated
          Critic          : Trained
          Critic_target   : Not Train, but updated
          Prediction during training
            1. In ENV
                Actor_target
            2. In Memory
                Critic Training : Q = R+max(Qt+1)
                  Q(t+1) = Critic(St+1,A1+1)
                  A(t+1) = Actor_target(St+1)
                Actor Training : Q = Critic(S,A)
                  A = Actor(S)
        '''
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # Critic Model
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.update_init()

    def create_noise(self,mu, theta, sigma):
        # mu = [mu of action1,mu of action2,... ]
        # theta = [theta of action1,theta of action2,... ]
        # sigma = [sigma of action1,sigma of action2,... ]
        for i in range(self.num_action):
            self.noise.append(OUNoise(mu[i], theta[i], sigma[i]))



    def act(self,state):
        self.ep *= self.epd
        if self.ep<=self.epmin:
            self.ep=self.epmin
        state1 = np.array(state[0]).reshape(1,self.num_state[0],self.window)
        hand = np.array(state[1]).reshape(1,self.num_state[1])

        if np.random.random() < self.ep:
            actlist = []
            sumact=0
            for i in range(self.num_action):
                actlist.append(random.randrange(1000)/1000)
                sumact+=actlist[-1]

            for i in range(self.num_action):
                actlist[i]/=sumact
            action = np.array([actlist]).reshape((1,self.num_action))
        else:
            action = self.actor_model.predict([state1,hand])

        self.update_num += 1


        return action



    def create_actor_model(self):
        stonk_input = tf.keras.layers.Input(shape=(self.num_state[0],self.window,))

        # Decoder
        de_x1 = tf.keras.layers.LSTM(self.a_nn[0],activation='tanh',return_sequences=True)(stonk_input)
        de_x1 = tf.keras.layers.Dropout(0.2)(de_x1)
        #de_x1 = tf.keras.layers.LayerNormalization()(de_x1)
        # Encoder
        en_x1 = tf.keras.layers.LSTM(self.a_nn[0],activation='tanh',return_sequences=True)(de_x1)
        en_x1 = tf.keras.layers.Dropout(0.2)(en_x1)
        #en_x1 = tf.keras.layers.LayerNormalization()(en_x1)
        # Attention layer [Recieve both decode and encode and add to decode]
        atten_x1 = tf.keras.layers.Attention()([en_x1, de_x1])
        # concate atten_x1 and de_x1
        atten_x1 = tf.keras.layers.concatenate([atten_x1,de_x1],axis=-1)
        # softmax function
        atten_x1 = tf.keras.layers.Flatten()(atten_x1)
        atten_x1 = tf.keras.layers.Dense(self.a_nn[0],activation='softmax')(atten_x1)

        #atten_x1 = tf.keras.layers.LayerNormalization()(atten_x1)

        hand_input = tf.keras.layers.Input(shape=tuple([self.num_state[1]]))
        x2 = tf.keras.layers.Dense(int(self.a_nn[0]),kernel_initializer=tf.keras.initializers.glorot_normal())(hand_input)
        x2 = tf.keras.layers.Activation('relu')(x2)

        x = tf.keras.layers.Concatenate()([atten_x1,x2])
        x = tf.keras.layers.LayerNormalization()(x)


        for i in range(len(self.a_nn)-1):
            x = tf.keras.layers.Dense(int(self.a_nn[i+1]),
                kernel_initializer=tf.keras.initializers.glorot_normal())(x)
            x = tf.keras.layers.Activation('relu')(x)

        # OUTPUT NODES
        output = tf.keras.layers.Dense(5,activation='softmax')(x)

        model = tf.keras.Model(inputs=[stonk_input,hand_input], outputs=output)
        model.compile(loss="huber", optimizer=tf.keras.optimizers.Adam(lr=self.lr*0.1)) # possion loss : count
        #model.summary()
        return stonk_input, model

    def create_critic_model(self):
        state_input = tf.keras.layers.Input(shape=(self.num_state[0],self.window,))

        # Decoder
        de_x1 = tf.keras.layers.LSTM(self.c_nn[0],activation='tanh',return_sequences=True)(state_input)
        de_x1 = tf.keras.layers.Dropout(0.2)(de_x1)
        #de_x1 = tf.keras.layers.LayerNormalization()(de_x1)
        # Encoder
        en_x1 = tf.keras.layers.LSTM(self.c_nn[0],activation='tanh',return_sequences=True)(de_x1)
        en_x1 = tf.keras.layers.Dropout(0.2)(en_x1)
        #en_x1 = tf.keras.layers.LayerNormalization()(en_x1)
        atten_x1 = tf.keras.layers.Attention()([en_x1, de_x1])
        # concate atten_x1 and de_x1
        atten_x1 = tf.keras.layers.concatenate([atten_x1,de_x1],axis=-1)
        # softmax function
        atten_x1 = tf.keras.layers.Flatten()(atten_x1)
        state_h1 = tf.keras.layers.Dense(self.c_nn[0],activation='softmax')(atten_x1)
        #state_h1 = tf.keras.layers.LayerNormalization()(state_h1)
        # ----------------------------------------------------------------------
        hand_input = tf.keras.layers.Input(shape=tuple([self.num_state[1]]))
        cx2 = tf.keras.layers.Dense(int(self.c_nn[0]),kernel_initializer=tf.keras.initializers.glorot_normal())(hand_input)
        cx2 = tf.keras.layers.Activation('relu')(cx2)

        action_input = tf.keras.layers.Input(shape=tuple([self.num_action]))
        act = tf.keras.layers.Dense(int(self.c_nn[0]),kernel_initializer=tf.keras.initializers.glorot_normal())(action_input)
        act = tf.keras.layers.Activation('relu')(act)


        x = tf.keras.layers.Concatenate()([state_h1, cx2, act])
        x = tf.keras.layers.LayerNormalization()(x)


        for i in range(len(self.c_nn)-1):
            x = tf.keras.layers.Dense(int(self.c_nn[i+1]),
                kernel_initializer=tf.keras.initializers.glorot_normal())(x)
            x = tf.keras.layers.Activation('relu')(x)

        output = tf.keras.layers.Dense(1)(x)

        model  = tf.keras.Model(inputs=[state_input,hand_input,action_input], outputs=output)
        model.compile(loss="huber", optimizer=tf.keras.optimizers.Adam(lr=self.lr,  clipnorm=1.)) # MAPE ( Mean Absolute Percentage Error)  : might be good for dataset with diff values
        #model.summary()
        return state_input, action_input, model

    def remember(self, state, action, reward, next_state, done):
        self.temprp.append([state, action, reward, next_state, done])

    def check_mem(self):
        try:
            if self.temprp[-1][3] == None:
                self.temprp.pop()
        except:
            pass



    #   Target Model Updating
    '''
    During training
      Actor           : Trained
      Actor_target    : Not Trained, but updated
      Critic          : Trained
      Critic_target   : Not Train, but updated
      Prediction during training
        1. In ENV
            Actor
        2. In Memory
            Critic Training : Q = R+max(Qt+1)
              Q(t+1) = Critic_target(St+1,A1+1)
              A(t+1) = Actor_target(St+1)
            Actor Training : Q = Critic(S,A)
              A = Actor(S)
    '''

    def train(self):
        '''
        New training function for actor-critic
        Train both Critic and Actor using tf.GradientTape()
        '''
        self.check_mem()
        # Batch

        batch_size = self.batch_size
        if len(self.temprp) < batch_size:
            return
        rewards = []

        samples = random.sample(self.temprp, batch_size)
        '''
        samples = []
        for num in range(int(self.batch_size/4)):
            random_no = random.randrange(0,len(self.temprp)-4+1)
            for i in range(4):
                samples.append(self.temprp[random_no+i])
        '''

        #print(len(samples))
        # Train Critc
        states1 = np.array([val[0][0] for val in samples])
        next_states1 = np.array([(np.zeros((self.num_state[0],self.window))
                                 if val[4] is 1 else val[3][0].reshape(self.num_state[0],self.window)) for val in samples])
        states2 = np.array([val[0][1] for val in samples])
        next_states2 = np.array([(np.zeros((1,self.num_state[1]))
                                 if val[4] is 1 else val[3][1].reshape(1,self.num_state[1])) for val in samples])
        #target_action = self.target_actor_model.predict_on_batch(states.reshape(-1,self.num_state))
        actions = np.array([val[1] for val in samples])
        # I dont know why  they cannot feed numpy into tensorflow
        states1=np.asarray(states1).astype(np.float32)
        next_states1 = np.asarray(next_states1).astype(np.float32)

        states2=np.asarray(states2).astype(np.float32)
        next_states2 = np.asarray(next_states2).astype(np.float32)

        q_s_a = self.critic_model.predict_on_batch([
            states1.reshape(-1,self.num_state[0],self.window),
            states2.reshape(-1,self.num_state[1]),
            actions.reshape(-1,self.num_action)])

        next_target_action = self.target_actor_model.predict_on_batch(
            [next_states1.reshape(-1,self.num_state[0],self.window),
            next_states2.reshape(-1,self.num_state[1])])

        q_s_a_d = self.target_critic_model.predict_on_batch([
            next_states1.reshape(-1,self.num_state[0],self.window),
            next_states2.reshape(-1,self.num_state[1]),
            next_target_action])

        # use target-q to calculate current_q and q_s_a_d
        x1 = np.zeros((len(samples), self.num_state[0],self.window))
        x2 = np.zeros((len(samples), self.num_state[1]))
        tar = np.zeros((len(samples), self.num_action))
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
            current_q = q_s_a[i]
            if done is 1:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward
            else:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward + self.gamma * np.amax(q_s_a_d[i])
            x1[i] = state[0].reshape(-1,self.num_state[0],self.window)
            x2[i] = state[1].reshape(-1,self.num_state[1])
            tar[i] = action.reshape(-1,self.num_action)
            y[i] = current_q

        c_loss = self.critic_model.train_on_batch([x1,x2, tar], y)
        #print('Loss {}'c_loss)
        # Using tf.GradientTape
        '''
          - [x,tar]->Critic->Q
          - LossC(y,Q)
          - x->Actor->A
          - [x,A]->Critic->Q
          - dQ/dA
        '''


        with tf.GradientTape() as tape:
            actor_preds = self.actor_model([x1,x2], training=True) # Prediction
            #tape.watch(actor_preds)
            critic_preds_2 = self.critic_model([x1,x2,
                                                actor_preds], training=True) # Prediction
            #tape.watch(critic_preds_2)
            actor_loss = -tf.math.reduce_mean(critic_preds_2)

        # Train Actor
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_weights)
        #keras.optimizers.Adam(lr=self.lr*0.1).apply_gradients(zip(actor_grads,self.actor_model.trainable_weights))
        tf.keras.optimizers.Adam(lr=self.lr*0.1,  clipnorm=1.).apply_gradients(zip(actor_grads,self.actor_model.trainable_weights))
    def _update_actor_target(self,init=None):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        if init==1:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = actor_model_weights[i]
        # Soft update using tau
        else:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = (actor_model_weights[i]*self.tau) + (actor_target_weights[i]*(1-self.tau))
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self,init=None):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        if init==1:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = critic_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = (critic_model_weights[i]*self.tau) + (critic_target_weights[i]*(1-self.tau))
        self.target_critic_model.set_weights(critic_target_weights) #use for train critic_model_weights

    def update(self):
        # Softupdate using tau every self.update_num interval
        if self.update_num == 600:
            self._update_actor_target()
            self._update_critic_target()
            self.update_num = 0
            print('update target')
        else:
            pass
        '''
        if self.update_lr == 150*60:
            self.lr = self.lr*0.75

            K.set_value(self.actor_model.optimizer.learning_rate, self.lr*0.1)
            K.set_value(self.target_actor_model.optimizer.learning_rate, self.lr*0.1)
            K.set_value(self.critic_model.optimizer.learning_rate, self.lr)
            K.set_value(self.target_critic_model.optimizer.learning_rate, self.lr)

            self.update_lr = 0
            print('update target')
        else:
            pass
        '''

    def update_init(self):
        self._update_actor_target(1)
        self._update_critic_target(1)
