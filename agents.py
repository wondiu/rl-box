# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers
from PIL import Image

from agent_networks import ActorNetwork, CriticNetwork, PreprocessingNetwork, PredictionNetwork, QNetwork
from noise import ParameterNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise

class DDPG_Agent():
    def __init__(self, sess, env, state_dim, n_features, n_actions, actionnable, noise,
                 gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3,
                 layer_norm=True, aux_pred=False, invert_gradients=False, optimism=0):
        self.sess = sess
        self.env = env
        self.state_dim = state_dim
        self.n_input = int(np.prod(state_dim))
        self.n_features = n_features
        self.n_actions = n_actions
        self.actionnable = actionnable
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.layer_norm = layer_norm
        self.param_noise_std = 1
        self.aux_pred = aux_pred
        
        self.record = []
        
        self.tau_ph = tf.placeholder(tf.float32, None)

        self.preprocess = PreprocessingNetwork(self.sess, 'Preprocess', self.state_dim, self.n_features, self.layer_norm)
        self.target_preprocess = PreprocessingNetwork(self.sess, 'TargetPreprocess', self.state_dim, self.n_features, self.layer_norm)
        self.target_preprocess.update_op = self.set_target_update(self.preprocess.vars, self.target_preprocess.vars)
        
        self.actor = ActorNetwork(self.sess, 'Actor', self.n_actions, None, self.preprocess, self.lr_actor, self.layer_norm, invert_gradients)
        self.target_actor = ActorNetwork(self.sess, 'TargetActor', self.n_actions, None, self.target_preprocess, self.lr_actor, self.layer_norm)
        self.target_actor.update_op = self.set_target_update(self.actor.vars, self.target_actor.vars)

        self.critic = CriticNetwork(self.sess, 'Critic', self.n_actions, None, self.preprocess, self.lr_critic, self.layer_norm, optimism)
        self.target_critic = CriticNetwork(self.sess, 'TargetCritic', self.n_actions, None, self.target_preprocess, self.lr_critic, self.layer_norm)
        self.target_critic.update_op = self.set_target_update(self.critic.vars, self.target_critic.vars)

        self.action_noise = None
        self.param_noise = None
        if noise['param'] is not None:
            self.param_noise = ParameterNoise(target_policy_std=noise['param'])
            self.param_noise_std_ph = tf.placeholder(tf.float32, ())
            self.perturbed_actor = ActorNetwork(self.sess, 'PerturbedActor', self.n_actions, None, self.preprocess, self.lr_actor, self.layer_norm)
            self.perturbed_actor.noise_vars = [tf.Variable(tf.random_normal(tf.shape(var), mean=0., stddev=1)) for var in self.perturbed_actor.pertubable_vars]
            self.perturbed_actor.update_op = self.set_perturb_update(self.actor.pertubable_vars, self.perturbed_actor.pertubable_vars, self.perturbed_actor.noise_vars)
            self.policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor.out - self.perturbed_actor.out)))
        if noise['norm'] is not None:
            self.action_noise = NormalActionNoise(mu=np.zeros(self.n_actions), sigma=noise['norm'])
        elif noise['OU'] is not None:
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_actions), sigma=noise['OU'])
        self.noise_decay = noise['decay']
            
        # prediction tasks
        if self.aux_pred:
            self.state_pred_task = PredictionNetwork(self.sess, 'StatePrediction', 
                                                     self.n_actions, None, self.preprocess,
                                                     self.n_input, learning_rate=1e-4, layer_norm=self.layer_norm)
            self.reward_pred_task = PredictionNetwork(self.sess, 'RewardPrediction', 
                                                     self.n_actions, None, self.preprocess,
                                                     1, learning_rate=1e-4, layer_norm=self.layer_norm)
            self.end_pred_task = PredictionNetwork(self.sess, 'EndPrediction', 
                                                     self.n_actions, None, self.preprocess,
                                                     1, learning_rate=1e-4, layer_norm=self.layer_norm,
                                                     classifier=True)    
    def reset_var(self, var):
        self.sess.run(var.initializer)

    def set_perturb_update(self, orig_vars, perturbed_vars, noise_vars):
        return tf.group(*[perturbed_vars[i].assign(orig_vars[i] + tf.multiply(noise_vars[i], self.param_noise_std_ph))
                    for i in range(len(perturbed_vars))])

    def set_target_update(self, orig_vars, target_vars):
        return tf.group(*[target_vars[i].assign(
                    tf.multiply(orig_vars[i], self.tau_ph) + tf.multiply(target_vars[i], 1. - self.tau_ph))
                    for i in range(len(target_vars))])
    
    def update_perturbed_net(self, std):
        for var in self.perturbed_actor.noise_vars:
            self.reset_var(var)
        self.sess.run(self.perturbed_actor.update_op, feed_dict={
             self.param_noise_std_ph: std,
        })       
    
    def update_target_nets(self, tau):
        self.sess.run([self.target_preprocess.update_op,
                       self.target_actor.update_op,
                       self.target_critic.update_op], feed_dict={
             self.tau_ph: tau,
        })
        
    def compute_policy_distance(self, inpt):
        return self.sess.run(self.policy_distance, feed_dict={
                self.actor.inpt: inpt,
                self.perturbed_actor.inpt: inpt
        })

    def init_step(self):
        self.update_target_nets(1)
    
    def restart_step(self):
        if self.param_noise is not None:
            self.update_perturbed_net(self.param_noise.std)
            self.param_noise.target_policy_std = max(0.01,
                                                     self.param_noise.target_policy_std*self.noise_decay)
        if self.action_noise is not None:
            self.action_noise.sigma =max(0.01,
                                         self.action_noise.sigma*self.noise_decay)
        
    def select_action(self, s): 
        if self.param_noise is not None:
            a = self.perturbed_actor.policy([s])[0]
        else:
            a = self.actor.policy([s])[0]
        if self.action_noise is not None:
            a += self.action_noise()
        a = [max(-1, min(1, a_i)) for a_i in a]
        if np.random.random() < 0.01:
            a = self.env.action_space.sample()
        return self.actionnable(a)
        
    def training_step(self, batch_size, replay_buffer):
        
        if replay_buffer.size() > batch_size:

            s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(batch_size)
            
            Q_targets = self.target_critic.compute_Q(s2_batch, self.target_actor.policy(s2_batch))
            ys = []
            for k in range(batch_size):
                if done_batch[k]:
                    ys.append(r_batch[k])
                else:
                    ys.append(r_batch[k] + self.gamma * Q_targets[k])
            ys = np.reshape(ys, (-1, 1))
            Q_values, loss, _ = self.critic.train(s_batch, a_batch, ys)
            new_a_batch = self.actor.policy(s_batch)
            a_gradients = self.critic.action_gradients(s_batch, new_a_batch)
            self.actor.train(s_batch, a_gradients[0], batch_size, new_a_batch)
            
#                    if total_iters%int(1/self.tau)==0:
            self.update_target_nets(self.tau)
            
            if self.param_noise is not None:
                self.param_noise.adapt(self.compute_policy_distance(s_batch))
                
            # Auxiliary tasks (state prediction, ...)
            if self.aux_pred:
                self.state_pred_task.train(s_batch, a_batch, s2_batch)
                self.reward_pred_task.train(s_batch, a_batch, np.reshape(r_batch, (-1, 1)))
                self.end_pred_task.train(s_batch, a_batch, np.reshape(done_batch, (-1, 1)).astype(float))

class DQN_Agent():
    def __init__(self, sess, env, state_dim, n_actions, dueling=True, optimism=0):
        self.sess = sess
        self.env = env
        self.state_dim = state_dim
        self.n_input = int(np.prod(state_dim))
        self.n_actions = n_actions
        self.optimism = 0
        
        self.epsilon = 1
        
        self.record = []
        
        self.tau_ph = tf.placeholder(tf.float32, None)
                    
        self.q_network = QNetwork(self.sess, 'Q-Network', self.n_actions,
                                self.state_dim, None, dueling=dueling, optimism=self.optimism)
        self.target_q_network = QNetwork(self.sess, 'TargetQ-Network', self.n_actions,
                                self.state_dim, None, dueling=dueling)
        self.target_q_network.update_op = self.set_target_update(self.q_network.vars,
                                                                 self.target_q_network.vars)
    
    def set_target_update(self, orig_vars, target_vars):
        return tf.group(*[target_vars[i].assign(
                    tf.multiply(orig_vars[i], self.tau_ph) + tf.multiply(target_vars[i], 1. - self.tau_ph))
                    for i in range(len(target_vars))])
        
    def update_target_net(self, tau):
        self.sess.run([self.target_q_network.update_op], feed_dict={
             self.tau_ph: tau,
        })
        
    def init_step(self):
        self.update_target_net(1)
            
    def epsilon_update(self, eps):
        self.epsilon = eps

    def select_action(self, s):
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = np.argmax(self.q_network.compute_Q([s])[0])           
        return a
        
    def training_step(self, batch_size, replay_buffer, gamma=0.99, lr=1e-4):
        
        if replay_buffer.size() > batch_size:

            s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(batch_size)
            
            Q_targets = np.max(self.target_q_network.compute_Q(s2_batch), 1)
            ys = []
            for k in range(batch_size):
                if done_batch[k]:
                    ys.append(r_batch[k])
                else:
                    ys.append(r_batch[k] + gamma * Q_targets[k])
#            print(self.q_network.compute_Q(s_batch)[0], ys[0])
            error, loss, _ = self.q_network.train(s_batch, a_batch, ys, lr)
            return loss

            
                            
