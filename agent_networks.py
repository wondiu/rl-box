# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers

class Network():
    def __init__(self, sess, name, inpt_shape, inpt_network=None):
        self.sess = sess
        self.name = name

        self.inpt_network = inpt_network

        with tf.variable_scope(self.name):
            if self.inpt_network is None:
                self.inpt_shape = inpt_shape
                self.inpt = tf.placeholder(tf.float32, (None,)+self.inpt_shape)
                self.out = self.build_network(self.inpt)
            else:
                self.inpt_shape = self.inpt_network.inpt.shape
                self.inpt = self.inpt_network.inpt
                self.out = self.build_network(self.inpt_network.out)
            
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.pertubable_vars = [var for var in self.trainable_vars if 'LayerNorm' not in var.name]
               
    def build_network(self, inpt):
        return inpt


class PreprocessingNetwork(Network):
    def __init__(self, sess, name, inpt_shape, n_features, layer_norm=True):
        self.n_features = n_features
        self.layer_norm = layer_norm
        super().__init__(sess=sess, name=name, inpt_shape=inpt_shape, inpt_network=None)
                       
    def build_network(self, inpt):
        h = inpt
#        h = layers.convolution2d(h, num_outputs=32, kernel_size=8, stride=4, activation_fn=None)
#        if self.layer_norm:
#            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
#        else:
#            h = tf.nn.relu(h)
#        h = layers.convolution2d(h, num_outputs=64, kernel_size=4, stride=2, activation_fn=None)
#        if self.layer_norm:
#            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
#        else:
#            h = tf.nn.relu(h)
#        h = layers.convolution2d(h, num_outputs=64, kernel_size=3, stride=1, activation_fn=None)
#        if self.layer_norm:
#            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
#        else:
#            h = tf.nn.relu(h)
#        h = layers.fully_connected(h, num_outputs=300, activation_fn=None)
#        if self.layer_norm:
#            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
#        else:
#            h = tf.nn.relu(h)

#        fs = layers.fully_connected(h, num_outputs=self.n_features, activation_fn=None)
#        if self.layer_norm:
#            fs = layers.layer_norm(fs, activation_fn=tf.nn.relu)
#        else:
#            fs = tf.nn.relu(fs)
#        return layers.flatten(fs)
        return inpt
    
class ActorNetwork(Network):
    def __init__(self, sess, name, n_actions, inpt_shape, inpt_network,
                 learning_rate=1e-4, layer_norm=True, invert_gradients=False):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.layer_norm = layer_norm
        self.invert_gradients = invert_gradients
        super().__init__(sess=sess, name=name, inpt_shape=inpt_shape, inpt_network=inpt_network)
        
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
        h = layers.fully_connected(h, num_outputs=300, activation_fn=None)
        if self.layer_norm:
            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
        else:
            h = tf.nn.relu(h)
        if not self.invert_gradients:
            actions = layers.fully_connected(h, num_outputs=self.n_actions,
                                             weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                             activation_fn=tf.nn.tanh)
        else:
            actions = layers.fully_connected(h, num_outputs=self.n_actions, activation_fn=None)
        return actions
           
    def train(self, inpt, a_gradients, batch_size, actions):
        if self.invert_gradients:
            def invert(grad, a):
                if grad<0:
                    return grad*(a+1)/2
                return grad*(1-a)/2
            for b in range(batch_size):
                a_gradients[b] = [invert(grad, a) for (grad, a) in zip(a_gradients[b], actions[b])]
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
    def __init__(self, sess, name, n_actions, inpt_shape, inpt_network, learning_rate=1e-3, layer_norm=True, optimism=0):
        self.sess = sess
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.layer_norm = layer_norm
        super().__init__(sess=sess, name=name, inpt_shape=inpt_shape, inpt_network=inpt_network)
         
        with tf.variable_scope(self.name):            
            self.ys = tf.placeholder(tf.float32, [None, 1])    
#            self.loss = tf.losses.mean_squared_error(self.ys, self.out)
            error = self.out - self.ys
            def aloss(a): return tf.pow(error, 2) * tf.pow(tf.sign(error) + a, 2)
            self.loss = aloss(-optimism)

            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.minimize(self.loss)
            self.a_gradients = tf.gradients(self.out, self.actions)

    def build_network(self, inpt):
        self.actions = tf.placeholder(tf.float32, [None, self.n_actions])
        h_a = self.actions
#        h_a = layers.fully_connected(h_a, num_outputs=32, activation_fn=None)
#        if self.layer_norm:
#            h_a = layers.layer_norm(h_a, activation_fn=tf.nn.relu)
#        else:
#            h_a = tf.nn.relu(h_a)
        h = tf.concat([inpt, h_a], 1)
#        h = tf.add(inpt, h_a)
        h = layers.fully_connected(h, num_outputs=300, activation_fn=None)
        if self.layer_norm:
            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
        else:
            h = tf.nn.relu(h)
        Q_values = layers.fully_connected(h, num_outputs=1, activation_fn=None)
        return Q_values

    def train(self, inpt, actions, ys):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
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
    
class QNetwork(Network):
    def __init__(self, sess, name, n_actions, inpt_shape, inpt_network,
                 learning_rate=1e-4, optimism=0):
        self.sess = sess
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        super().__init__(sess=sess, name=name, inpt_shape=inpt_shape, inpt_network=inpt_network)
         
        with tf.variable_scope(self.name):            
            self.ys = tf.placeholder(tf.float32, [None])
            self.actions = tf.placeholder(tf.int32, [None])
            self.selected_out = tf.reduce_sum(tf.multiply(self.out, tf.one_hot(self.actions, self.n_actions)), 1)
            error = self.ys - self.selected_out
            def aloss(a): return tf.pow(error, 2) * tf.pow(tf.sign(error) + a, 2)
            self.loss = aloss(-optimism)

            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.minimize(self.loss)

    def build_network(self, inpt):
        h = layers.fully_connected(inpt, num_outputs=300, activation_fn=tf.nn.relu)
        h = layers.fully_connected(h, num_outputs=300, activation_fn=tf.nn.relu)
        out = layers.fully_connected(h, num_outputs=self.n_actions, activation_fn=None)
        return out

    def train(self, inpt, actions, ys):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inpt: inpt,
            self.actions: actions,
            self.ys: ys
        })

    def compute_Q(self, inpt):
        return self.sess.run(self.out, feed_dict={
            self.inpt: inpt
        })

    def compute_selected_Q(self, inpt, actions):
        return self.sess.run(self.selected_out, feed_dict={
            self.inpt: inpt,
            self.actions: actions
        })

    
class PredictionNetwork(Network):
    def __init__(self, sess, name, n_actions, inpt_shape, inpt_network, n_output,
                 learning_rate=1e-4, layer_norm=True, classifier=False):
        self.sess = sess
        self.n_actions = n_actions
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.layer_norm = layer_norm
        self.classifier = classifier
        super().__init__(sess=sess, name=name, inpt_shape=inpt_shape, inpt_network=inpt_network)
         
        with tf.variable_scope(self.name):            
            self.ys = tf.placeholder(tf.float32, [None, self.n_output])
            if self.classifier:
                self.loss = tf.losses.sigmoid_cross_entropy(self.ys, self.out)/self.n_output
            else:
                self.loss = tf.losses.mean_squared_error(self.ys, self.out)/self.n_output
            self.trainer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize = self.trainer.minimize(self.loss)

    def build_network(self, inpt):
        self.actions = tf.placeholder(tf.float32, [None, self.n_actions])
        h_a = self.actions
#        h_a = layers.fully_connected(h_a, num_outputs=32, activation_fn=None)
#        if self.layer_norm:
#            h_a = layers.layer_norm(h_a, activation_fn=tf.nn.relu)
#        else:
#            h_a = tf.nn.relu(h_a)
        h = tf.concat([inpt, h_a], 1)
#        h = tf.add(inpt, h_a)
        h = layers.fully_connected(h, num_outputs=64, activation_fn=None)
        if self.layer_norm:
            h = layers.layer_norm(h, activation_fn=tf.nn.relu)
        else:
            h = tf.nn.relu(h)
        pred = layers.fully_connected(h, num_outputs=self.n_output, activation_fn=None)
        return pred

    def train(self, inpt, actions, ys):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inpt: inpt,
            self.actions: actions,
            self.ys: ys
        })

    def predict(self, inpt, actions):
        return self.sess.run(self.out, feed_dict={
            self.inpt: inpt,
            self.actions: actions
        })
