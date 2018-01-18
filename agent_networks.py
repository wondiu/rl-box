# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

class PerceptionNetwork():
    def __init__(self, sess, n_input, n_features):
        self.sess = sess
        self.n_input = n_input
        self.n_features = n_features

        with tf.variable_scope('Percept'):
            self.s = tf.placeholder(tf.float32, [None, self.n_input], "state")
    
            h = self.s
            h = layers.fully_connected(h, num_outputs=20, activation_fn=tf.nn.relu)
        
            self.fs = layers.fully_connected(h, num_outputs=self.n_features, activation_fn=tf.nn.relu)

    
class ActorNetwork():
    def __init__(self, sess, n_actions, perception_net, tau=1e-3, learning_rate=1e-4):
        self.sess = sess
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.tau = tau
        self.perception_net = perception_net
        
        with tf.variable_scope('Actor'):
            self.inpt = self.perception_net.fs
            self.act_probs, self.a  = self.build_network(self.inpt)

            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
            
            self.action_gradients = tf.placeholder(tf.float32, [None, self.n_actions])
            self.batch_size = tf.placeholder(tf.float32, None)
    
            self.actor_gradients = tf.gradients(self.act_probs, self.network_params, -self.action_gradients)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.actor_gradients))
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.apply_gradients(zip(self.actor_gradients, self.network_params))
    
        with tf.variable_scope('ActorTarget'):
            self.target_act_probs, self.target_a = self.build_network(self.inpt)
    
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ActorTarget')
    
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                    for i in range(len(self.target_network_params))]

    def build_network(self, inpt):
        h = inpt
#        h = layers.fully_connected(h, num_outputs=10, activation_fn=tf.nn.relu)

        act_probs = layers.fully_connected(h, num_outputs=self.n_actions, activation_fn=tf.nn.softmax)
        a = tf.argmax(act_probs, axis=1)
        return act_probs, a
           
    def train(self, inpt, a_gradients, batch_size):
        self.sess.run(self.optimize, feed_dict={
            self.perception_net.s: inpt,
            self.action_gradients: a_gradients,
            self.batch_size: batch_size
        })

    def policy(self, inpt):
        return self.sess.run(self.act_probs, feed_dict={
             self.perception_net.s: inpt,
        })

    def choose_action(self, inpt):
        return self.sess.run(self.a, feed_dict={
                self.perception_net.s: inpt})
        
    def target_choose_action(self, inpt):
        return self.sess.run(self.target_a, feed_dict={
                self.perception_net.s: inpt})
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    
class CriticNetwork():
    def __init__(self, sess, n_actions, perception_net, tau=1e-3, learning_rate=1e-3):
        self.sess = sess
        self.n_actions = n_actions
        self.tau = tau
        self.learning_rate = learning_rate
        self.perception_net = perception_net
         
        with tf.variable_scope('Critic'):
            self.action = tf.placeholder(tf.float32, [None, self.n_actions])
            self.inpt = tf.concat([self.perception_net.fs, self.action], 1)
            self.Q_values = self.build_network(self.inpt)
 
            self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
            
            self.ys = tf.placeholder(tf.float32, [None, 1])    
            self.loss = tf.losses.mean_squared_error(self.ys, self.Q_values)
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.minimize(self.loss)
    
            self.a_gradients = tf.gradients(self.Q_values, self.action)

        with tf.variable_scope('CriticTarget'):
            self.target_Q_values = self.build_network(self.inpt)
    
            self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CriticTarget')
    
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                    for i in range(len(self.target_network_params))]


    def build_network(self, inpt):
        h = self.inpt
#        h = layers.fully_connected(h, num_outputs=10, activation_fn=tf.nn.relu)
        Q_values = layers.fully_connected(h, num_outputs=1, activation_fn=None)
        return Q_values

    def train(self, inpt, action, ys):
        return self.sess.run([self.Q_values, self.optimize], feed_dict={
            self.perception_net.s: inpt,
            self.action: action,
            self.ys: ys
        })

    def compute_Q(self, inpt, action):
        return self.sess.run(self.Q_values, feed_dict={
            self.perception_net.s: inpt,
            self.action: action
        })

    def action_gradients(self, inpt, action):
        return self.sess.run(self.a_gradients, feed_dict={
            self.perception_net.s: inpt,
            self.action: action
        })

    def target_compute_Q(self, inpt, action):
        return self.sess.run(self.target_Q_values, feed_dict={
            self.perception_net.s: inpt,
            self.action: action
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)