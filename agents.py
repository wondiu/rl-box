# -*- coding: utf-8 -*-

from copy import copy
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

from replay_buffer import ReplayBuffer
from agent_networks import ActorNetwork, CriticNetwork, PreprocessingNetwork


def copy_network(net, name):
    copied_net = copy(net)
    copied_net.name = name
    return copied_net

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
    def __init__(self, sess, env, n_input, n_features, n_actions, action_bounds, discrete,
                 gamma=0.99, tau=1e-3, buffer_size=1e5, lr_actor=1e-4, lr_critic=1e-3, layer_norm=True):
        self.sess = sess
        self.env = env
        self.n_input = n_input
        self.n_features = n_features
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        self.discrete = discrete
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.layer_norm = layer_norm
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.record = []

        self.preprocess = PreprocessingNetwork(self.sess, 'Preprocess', self.n_input, self.n_features, self.layer_norm)
        self.target_preprocess = copy_network(self.preprocess, 'TargetPreprocess')
        self.target_preprocess.update_op = self.set_target_update(self.preprocess.vars, self.target_preprocess.vars, self.tau)
        
        
        self.actor = ActorNetwork(self.sess, 'Actor', self.n_actions, None, self.preprocess, self.lr_actor, self.layer_norm)
        self.target_actor = copy_network(self.actor, 'TargetActor')
        self.target_actor.update_op = self.set_target_update(self.actor.vars, self.target_actor.vars, self.tau)


        self.critic = CriticNetwork(self.sess, 'Critic', self.n_actions, None, self.preprocess, self.lr_critic, self.layer_norm)
        self.target_critic = copy_network(self.critic, 'TargetCritic')
        self.target_critic.update_op = self.set_target_update(self.critic.vars, self.target_critic.vars, self.tau)

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_actions))
        
    def set_target_update(self, orig_vars, target_vars, tau):
        return tf.group(*[target_vars[i].assign(
                    tf.multiply(orig_vars[i], self.tau) + tf.multiply(target_vars[i], 1. - self.tau))
                    for i in range(len(target_vars))])

    def update_target_network(self, target_network):
        self.sess.run(target_network.update_op)

    
    def actionnable(self, a):
        if self.discrete:
            return np.array([int(a_i<0) for a_i in a])[0] #HACK
        else:
            return a*self.action_bounds

        
    def train(self, max_episodes, max_episode_len, batch_size, render=False):
        for i in range(max_episodes):    
            s = self.env.reset()
    
            ep_reward = 0
            ep_ave_max_q = 0
    
            for j in range(max_episode_len):
                if render:
                    self.env.render()
    
                a = self.actor.policy([s])[0] + self.actor_noise()

                s2, r, done, info = self.env.step(self.actionnable(a))
                
                self.replay_buffer.add( (s, a, r, done, s2) )

                if self.replay_buffer.size() > batch_size:
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)
                    
                    Q_targets = self.target_critic.compute_Q(s2_batch, self.target_actor.policy(s2_batch))
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
        
                    self.update_target_network(self.target_preprocess)
                    self.update_target_network(self.target_actor)
                    self.update_target_network(self.target_critic)

                s = s2
                ep_reward += r

                if done:
                    break
            if i%1==0:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))

            self.record.append(ep_reward)

