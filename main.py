# -*- coding: utf-8 -*-

import numpy as np
import random
import tensorflow as tf
import gym
import time

from agents import DDPG_Agent
from wrappers import make_atari, wrap_deepmind

def actionnable(a):
    if discrete:
        return np.array([int(a_i<0) for a_i in a])[0] #HACK
    else:
        return a*self.action_bounds

def actionnable_breakout(a):
    return np.array([int(a_i<0)+2 for a_i in a])[0] # DOUBLE HACK


envs = ['Pendulum-v0', 'CartPole-v0', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
env_name = envs[2]
random_seed = 124
max_episodes = 10000
max_episode_len = 10000
render = False
batch_size = 32
learning_freq = 4
gamma = 0.99
tau = 1e-3
layer_norm = True
noise = {'type':'param', 'std':0.5} # type = param, norm, OU or none

R=[]

def main():
    with tf.Session() as sess:

        
        if env_name == 'CartPole-v0':
            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = 1
            action_bounds = [1]
        elif env_name == 'Pendulum-v0':
            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            action_bounds = env.action_space.high
            assert (env.action_space.high == -env.action_space.low)
        elif env_name == 'BreakoutNoFrameskip-v4':
            env = wrap_deepmind(make_atari(env_name), episode_life=True, clip_rewards=True, frame_stack=True)
            state_dim = env.observation_space.shape
            action_dim = 1 #env.action_space.n
            print(env.unwrapped.get_action_meanings())
            action_bounds = [1]         
            
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)

           
        agent = DDPG_Agent(sess, env, state_dim, 128, action_dim, action_bounds, actionnable_breakout,
                           gamma, tau, lr_actor=1e-4, lr_critic=1e-3, layer_norm=layer_norm,
                           noise=noise)
        
        sess.run(tf.global_variables_initializer())
            
        agent.train(max_episodes, max_episode_len, batch_size, learning_freq, render)
        
        env.close()
        return agent.record

if __name__ == '__main__':
    
    tf.reset_default_graph()
    t0 = time.time()
    R=main()
    print(time.time()-t0)

