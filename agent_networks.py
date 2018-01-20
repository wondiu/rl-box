# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Network():
    def __init__(self, sess, name, n_input, inpt_network=None):
        self.sess = sess
        self.name = name
        self.n_input = n_input
        self.inpt_network = inpt_network

        with tf.variable_scope(self.name):
            if self.inpt_network is None:
                self.inpt = tf.placeholder(tf.float32, [None, self.n_input])
                self.out = self.build_network(self.inpt)
            else:
                self.inpt = self.inpt_network.inpt
                self.out = self.build_network(self.inpt_network.out)
            
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.pertubable_vars = [var for var in self.trainable_vars if 'LayerNorm' not in var.name]
               
    def build_network(self, inpt):
        return inpt


class PreprocessingNetwork(Network):
    def __init__(self, sess, name, n_input, n_features, layer_norm=True):
        self.n_features = n_features
        self.layer_norm = layer_norm
        super().__init__(sess=sess, name=name, n_input=n_input, inpt_network=None)
        
        self.s= self.inpt
        self.fs = self.out

               
    def build_network(self, inpt):
        h = inpt
#        h = layers.fully_connected(h, num_outputs=20, activation_fn=None)
#        if self.layer_norm:
#            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
#        else:
#            h = tf.nn.relu(h)
        fs = layers.fully_connected(h, num_outputs=self.n_features, activation_fn=None)
        if self.layer_norm:
            fs = layers.layer_norm(fs, activation_fn=tf.nn.relu)
        else:
            fs = tf.nn.relu(fs)
        return fs
    
class ActorNetwork(Network):
    def __init__(self, sess, name, n_actions, n_input, inpt_network, learning_rate=1e-3, layer_norm=True):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.layer_norm = layer_norm
        super().__init__(sess=sess, name=name, n_input=n_input, inpt_network=inpt_network)
        
        with tf.variable_scope(self.name):            
            self.action_gradients = tf.placeholder(tf.float32, [None, self.n_actions])
            self.batch_size = tf.placeholder(tf.float32, None)
            
            self.total_trainable_vars = self.trainable_vars
            if inpt_network is not None:
                self.total_trainable_vars += inpt_network.trainable_vars
    
            self.actor_gradients = tf.gradients(self.out, self.total_trainable_vars, -self.action_gradients)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.actor_gradients))
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.apply_gradients(zip(self.actor_gradients, self.total_trainable_vars))

    def build_network(self, inpt):
        h = inpt
        h = layers.fully_connected(h, num_outputs=16, activation_fn=None)
        if self.layer_norm:
            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
        else:
            h = tf.nn.relu(h)
            
        actions = layers.fully_connected(h, num_outputs=self.n_actions,
                                         weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                         activation_fn=tf.nn.tanh)
        return actions
           
    def train(self, inpt, a_gradients, batch_size):
        self.sess.run(self.optimize, feed_dict={
            self.inpt: inpt,
            self.action_gradients: a_gradients,
            self.batch_size: batch_size
        })

    def policy(self, inpt):
        return self.sess.run(self.out, feed_dict={
             self.inpt: inpt,
        })
    
class CriticNetwork(Network):
    def __init__(self, sess, name, n_actions, n_input, inpt_network, learning_rate=1e-4, layer_norm=True):
        self.sess = sess
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.layer_norm = layer_norm
        super().__init__(sess=sess, name=name, n_input=n_input, inpt_network=inpt_network)
         
        with tf.variable_scope(self.name):            
            self.ys = tf.placeholder(tf.float32, [None, 1])    
            self.loss = tf.losses.mean_squared_error(self.ys, self.out)
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.minimize(self.loss)
            self.a_gradients = tf.gradients(self.out, self.actions)

    def build_network(self, inpt):
        self.actions = tf.placeholder(tf.float32, [None, self.n_actions])
        h = tf.concat([inpt, self.actions], 1)
        h = layers.fully_connected(h, num_outputs=16, activation_fn=None)
        if self.layer_norm:
            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
        else:
            h = tf.nn.relu(h)
        Q_values = layers.fully_connected(h, num_outputs=1, activation_fn=None)
        return Q_values

    def train(self, inpt, actions, ys):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inpt: inpt,
            self.actions: actions,
            self.ys: ys
        })

    def compute_Q(self, inpt, actions):
        return self.sess.run(self.out, feed_dict={
            self.inpt: inpt,
            self.actions: actions
        })

    def action_gradients(self, inpt, actions):
        return self.sess.run(self.a_gradients, feed_dict={
            self.inpt: inpt,
            self.actions: actions
        })