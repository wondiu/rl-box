# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import gym
import time
import matplotlib.pyplot as plt

from agents import DDPG_Agent
from wrappers import make_atari, wrap_deepmind

discrete_envs = ['CartPole-v0']
continuous_envs = ['Pendulum-v0', 'LunarLanderContinuous-v2', 'BipedalWalker-v2']
atari4_envs =['BreakoutNoFrameskip-v4']
env_name = continuous_envs[1]
random_seed = 121354
max_episodes = 500
max_episode_len = 1005

render = False
batch_size = 32
learning_freq = 1
gamma = 0.99
tau = 1e-3
layer_norm = True
noise = {'norm':None, 'OU':None, 'param':None}

R=[]

def main():
    with tf.Session() as sess:
        if env_name in discrete_envs:
            env = gym.make(env_name)
            state_dim = env.observation_space.shape
            action_dim = 1
            def actionnable(a):
                return np.array([int(a_i<0) for a_i in a])[0] #HACK
        elif env_name in continuous_envs:
            env = gym.make(env_name)
            state_dim = env.observation_space.shape
            action_dim = env.action_space.shape[0]
            action_bounds = env.action_space.high
            def actionnable(a):
                return a*action_bounds
        elif env_name in atari4_envs:
            env = wrap_deepmind(make_atari(env_name), episode_life=True, clip_rewards=True, frame_stack=True)
            state_dim = env.observation_space.shape
            action_dim = 1 #env.action_space.n
            print(env.unwrapped.get_action_meanings())
            action_bounds = [1]         
            def actionnable(a):
                return np.array([int(a_i<0)+2 for a_i in a])[0] # DOUBLE HACK
            
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)

           
        agent = DDPG_Agent(sess, env, state_dim, 64, action_dim, actionnable,
                           gamma, tau, lr_actor=1e-4, lr_critic=1e-3, layer_norm=layer_norm,
                           noise=noise)
        
        sess.run(tf.global_variables_initializer())
            
        agent.train(max_episodes, max_episode_len, batch_size, learning_freq, render)
        
        env.close()
        return agent.record

if __name__ == '__main__':
    
    tf.reset_default_graph()
    t0 = time.time()
    R =main()
    print(time.time()-t0)

