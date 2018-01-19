# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

from replay_buffer import ReplayBuffer
from agent_networks import ActorNetwork, CriticNetwork, PerceptionNetwork

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class DDPG_Agent():
    def __init__(self, sess, env, n_input, n_features, n_actions, action_bound,
                 gamma=0.99, tau=1e-3, buffer_size=1e5, lr_actor=1e-4, lr_critic=1e-3):
        self.sess = sess
        self.env = env
        self.n_input = n_input
        self.n_features = n_features
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.record = []

        self.percept = PerceptionNetwork(self.sess, self.n_input, self.n_features)
        self.actor = ActorNetwork(self.sess, self.n_actions, self.action_bound, self.percept, self.tau, lr_actor)
        self.critic = CriticNetwork(self.sess, self.n_actions, self.percept, self.tau, lr_critic)
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_actions))
        
    def train(self, max_episodes, max_episode_len, batch_size, render=False):
        for i in range(max_episodes):    
            s = self.env.reset()
    
            ep_reward = 0
            ep_ave_max_q = 0
    
            for j in range(max_episode_len):
                if render:
                    self.env.render()
    
                a = self.actor.policy([s])[0] + self.actor_noise()
                s2, r, done, info = self.env.step(a)
                
                self.replay_buffer.add( (s, a, r, done, s2) )

                if self.replay_buffer.size() > batch_size:
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)
                    
                    Q_targets = self.critic.target_compute_Q(s2_batch, self.actor.target_policy(s2_batch))
                    ys = []
                    for k in range(batch_size):
                        if done_batch[k]:
                            ys.append(r_batch[k])
                        else:
                            ys.append(r_batch[k] + self.gamma * Q_targets[k])
        
                    Q_values, _ = self.critic.train(s_batch, a_batch, np.reshape(ys, (-1, 1)))
                    ep_ave_max_q += np.amax(Q_values)
        
                    a_gradients = self.critic.action_gradients(s_batch, self.actor.policy(s_batch))
                    
                    self.actor.train(s_batch, a_gradients[0], batch_size)
        
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

