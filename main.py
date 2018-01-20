# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym

from agents import DDPG_Agent


env_name = 'Pendulum-v0'
#env_name = 'CartPole-v0'
random_seed = 4512
max_episodes = 500
max_episode_len = 1000
render = False
batch_size = 128
gamma = 0.99
tau = 1e-3
layer_norm = True

Rnorm=[]
R=[]


def main():
    with tf.Session() as sess:

        env = gym.make(env_name)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)

        state_dim = env.observation_space.shape[0]
        
        if env_name == 'CartPole-v0':
            action_dim = 1
            action_bounds = [1]
            discrete = True
        else:
            action_dim = env.action_space.shape[0]
            action_bounds = env.action_space.high
            assert (env.action_space.high == -env.action_space.low)
            discrete = False
           
        agent = DDPG_Agent(sess, env, state_dim, 32, action_dim, action_bounds, discrete,
                           gamma, tau, lr_actor=1e-4, lr_critic=1e-3, layer_norm=layer_norm)
        
        sess.run(tf.global_variables_initializer())
            

        agent.train(max_episodes, max_episode_len, batch_size, False)
        return agent.record

if __name__ == '__main__':
    
    tf.reset_default_graph()
    R=main()

