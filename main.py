# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym

from agents import DDPG_Agent


env_name = 'Pendulum-v0'
random_seed = 1245
max_episodes = 1000
max_episode_len = 1000
render = False
batch_size = 64
gamma = 0.99
epsilon = 1
tau = 1e-3

R=[]


def main():
    with tf.Session() as sess:

        env = gym.make(env_name)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        assert (env.action_space.high == -env.action_space.low)
       
        agent = DDPG_Agent(sess, env, state_dim, 20, action_dim, action_bound, gamma, tau)
        
        sess.run(tf.global_variables_initializer())
            

        agent.train(max_episodes, max_episode_len, batch_size, False)
        return agent.record

if __name__ == '__main__':
    
    tf.reset_default_graph()
    R=main()
