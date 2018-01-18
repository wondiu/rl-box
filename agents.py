# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

from replay_buffer import ReplayBuffer
from agent_networks import ActorNetwork, CriticNetwork, PerceptionNetwork

def one_hot(i, n):
    return np.eye(n)[i]

class DDPG_Agent():
    def __init__(self, sess, env, n_input, n_features, n_actions, epsilon=0.1, gamma=0.99, buffer_size=1e3):
        self.sess = sess
        self.env = env
        self.n_input = n_input
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.record = []

        self.percept = PerceptionNetwork(self.sess, self.n_input, self.n_features)
        self.actor = ActorNetwork(self.sess, self.n_actions, self.percept)
        self.critic = CriticNetwork(self.sess, self.n_actions, self.percept)
        
    def train(self, max_episodes, max_episode_len, batch_size, render=False):
        for i in range(max_episodes):
    
            s = self.env.reset()
    
            ep_reward = 0
            ep_ave_max_q = 0
    
            for j in range(max_episode_len):
                if render:
                    self.env.render()
    
                if np.random.random()<self.epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = self.actor.choose_action([s])[0]                    
                a_one_hot = one_hot(a, self.n_actions)
                s2, r, done, info = self.env.step(a)
                if i%100==0:
                    print(self.actor.policy([s]), a)
                if done: r=-1
                
                self.replay_buffer.add( (s, a_one_hot, r, done, s2) )

                if self.replay_buffer.size() > batch_size:
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)
                    
                    Q_targets = self.critic.target_compute_Q(s2_batch, one_hot(self.actor.target_choose_action(s2_batch), self.n_actions))
                    ys = []
                    for k in range(batch_size):
                        if done_batch[k]:
                            ys.append(r_batch[k])
                        else:
                            ys.append(r_batch[k] + self.gamma * Q_targets[k])
        
                    Q_values, _ = self.critic.train(s_batch, a_batch, np.reshape(ys, (-1, 1)))
                    ep_ave_max_q += np.amax(Q_values)
        
                    a_gradients = self.critic.action_gradients(s_batch, self.actor.policy(s_batch))[0]
                    self.actor.train(s_batch, a_gradients, batch_size)
        
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                s = s2
                ep_reward += r
    
                if done:
                    if i%10==0:
                        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                i, (ep_ave_max_q / float(j))))
                    self.record.append(ep_reward)
                    break

