# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

from replay_buffer import ReplayBuffer
from agent_networks import ActorNetwork, CriticNetwork, PreprocessingNetwork
from noise import ParameterNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise

class DDPG_Agent():
    def __init__(self, sess, env, state_dim, n_features, n_actions, action_bounds, actionnable,
                 gamma=0.99, tau=1e-3, buffer_size=1e5, lr_actor=1e-4, lr_critic=1e-3,
                 layer_norm=True, noise={'type':'param', 'std':0.2}):
        self.sess = sess
        self.env = env
        self.state_dim = state_dim
        self.n_features = n_features
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        self.actionnable = actionnable
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.layer_norm = layer_norm
        self.param_noise_std = 1
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.record = []
        
        self.tau_ph = tf.placeholder(tf.float32, None)

        self.preprocess = PreprocessingNetwork(self.sess, 'Preprocess', self.state_dim, self.n_features, self.layer_norm)
        self.target_preprocess = PreprocessingNetwork(self.sess, 'TargetPreprocess', self.state_dim, self.n_features, self.layer_norm)
        self.target_preprocess.update_op = self.set_target_update(self.preprocess.vars, self.target_preprocess.vars)
        
        self.actor = ActorNetwork(self.sess, 'Actor', self.n_actions, None, self.preprocess, self.lr_actor, self.layer_norm)
        self.target_actor = ActorNetwork(self.sess, 'TargetActor', self.n_actions, None, self.target_preprocess, self.lr_actor, self.layer_norm)
        self.target_actor.update_op = self.set_target_update(self.actor.vars, self.target_actor.vars)

        self.critic = CriticNetwork(self.sess, 'Critic', self.n_actions, None, self.preprocess, self.lr_critic, self.layer_norm)
        self.target_critic = CriticNetwork(self.sess, 'TargetCritic', self.n_actions, None, self.target_preprocess, self.lr_critic, self.layer_norm)
        self.target_critic.update_op = self.set_target_update(self.critic.vars, self.target_critic.vars)

        self.action_noise = None
        self.param_noise = None
        if noise['type'] == 'param':
            self.param_noise = ParameterNoise(target_policy_std=noise['std'])
            self.param_noise_std_ph = tf.placeholder(tf.float32, ())
            self.perturbed_actor = ActorNetwork(self.sess, 'PerturbedActor', self.n_actions, None, self.preprocess, self.lr_actor, self.layer_norm)
            self.perturbed_actor.noise_vars = [tf.Variable(tf.random_normal(tf.shape(var), mean=0., stddev=1)) for var in self.perturbed_actor.pertubable_vars]
            self.perturbed_actor.update_op = self.set_perturb_update(self.actor.pertubable_vars, self.perturbed_actor.pertubable_vars, self.perturbed_actor.noise_vars)
            self.policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor.out - self.perturbed_actor.out)))
        elif noise['type'] == 'norm':
            self.action_noise = NormalActionNoise(mu=np.zeros(self.n_actions), sigma=noise['std'])
        elif noise['type'] == 'OU':
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_actions), sigma=noise['std'])
        
    def test(self):
        return self.sess.run(self.perturbation)
    
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
        
    def train(self, max_episodes, max_episode_len, batch_size, learning_freq=1, render=False):
        # Hard updates for initialisation
        self.update_target_nets(1)
        ep_rewards = deque(maxlen = 100)
        total_iters = 0
        for i in range(max_episodes):
            s = self.env.reset()
            
            ep_reward = 0
            for j in range(max_episode_len):
                total_iters += 1
                if render:
                    self.env.render()
                if self.param_noise is not None:
                    self.update_perturbed_net(self.param_noise.std)
                    a = self.perturbed_actor.policy([s])[0]
                elif self.action_noise is not None:
                    a = self.actor.policy([s])[0] + self.action_noise()
                    a = [max(-1, min(1, a_i)) for a_i in a]
                else:
                    a = self.actor.policy([s])[0]

                s2, r, done, info = self.env.step(self.actionnable(a))
                
                self.replay_buffer.add( (s, a, r, done, s2) )

                if self.replay_buffer.size() > batch_size and j%learning_freq==0:
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(batch_size)
                    
                    Q_targets = self.target_critic.compute_Q(s2_batch, self.target_actor.policy(s2_batch))
                    ys = []
                    for k in range(batch_size):
                        if done_batch[k]:
                            ys.append(r_batch[k])
                        else:
                            ys.append(r_batch[k] + self.gamma * Q_targets[k])
                    ys = np.reshape(ys, (-1, 1))
                    Q_values, _ = self.critic.train(s_batch, a_batch, ys)
                    a_gradients = self.critic.action_gradients(s_batch, self.actor.policy(s_batch))
                    self.actor.train(s_batch, a_gradients[0], batch_size)
        
                    self.update_target_nets(self.tau)
                    
                    if self.param_noise is not None:
                        self.param_noise.adapt(self.compute_policy_distance(s_batch))
                        
                s = s2
                ep_reward += r

                if done:
                    break
            ep_rewards.append(ep_reward)
            if i%1==0:
                print('| Episode: {:d} | Length: {:d} | Reward: {:d} | Running: {:f} | '.format(i, j, int(ep_reward), np.mean(ep_rewards))+str(a))

            self.record.append(np.mean(ep_rewards))
        print(total_iters)

