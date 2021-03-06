# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import shutil
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import time
import matplotlib.pyplot as plt
from PIL import Image

from agents import DDPG_Agent, DQN_Agent
from replay_buffer import ReplayBuffer, SplitBuffer
from trainer import Trainer
import wrappers

discrete_envs = ['CartPole-v0', 'MountainCar-v0', 'LunarLander-v2', 'Pong-ram-v0']
continuous_envs = ['Pendulum-v0', 'MountainCarContinuous-v0', 'LunarLanderContinuous-v2',
                   'BipedalWalker-v2', 'BipedalWalkerHardcore-v2']
atari4_envs =['BreakoutNoFrameskip-v4']
env_name = discrete_envs[2]
random_seed = 1486
max_episodes = 400
max_time = 60*60

render = False
batch_size = 64


learning_freq = 1
gamma = 0.99
tau = 1e-3
buffer_size = 2e5
layer_norm = False
invert_gradients = False
noise = {'norm':None, 'OU':None, 'param':None, 'decay':0.99}
aux_pred = False
optimism = 0
repeat_action = 1

load = False
save = True
load_dir = "./save_dir"
save_dir = "./save_dir"

cpu_only = False
if cpu_only:
    config = tf.ConfigProto(
            device_count = {'GPU': 0})
else:
    config = tf.ConfigProto()

replay_buffer = ReplayBuffer(buffer_size)

def main():
    with tf.Session(config=config) as sess:
        if env_name in discrete_envs:
            env = gym.make(env_name)
#            env = wrappers.NoopResetEnv(env)
#            env = wrappers.EpisodicLifeEnv(env)
#            env = wrappers.FireResetEnv(env)
#            if clip:
#                env = wrappers.ClipRewardEnv(env)

            state_dim = env.observation_space.shape
            action_dim = env.action_space.n
            def actionnable(a):
                return np.array([int(a_i<0) for a_i in a])[0] #HACK
        elif env_name in continuous_envs:
            env = gym.make(env_name)
            action_dim = env.action_space.shape[0]
            state_dim = (env.observation_space.shape[0],)
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
            
        print(state_dim, action_dim)
            
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        env.seed(random_seed)
        
#        agent = DDPG_Agent(sess, env, state_dim, 300, action_dim, actionnable, noise,
#                           gamma, tau, lr_actor=1e-4, lr_critic=1e-3,
#                           layer_norm=layer_norm,
#                           aux_pred=aux_pred, invert_gradients=invert_gradients,
#                           optimism=optimism)
        
        agent = DQN_Agent(sess, env, state_dim, action_dim,
                          dueling=True, optimism=optimism)
        
        saver = tf.train.Saver()
        
        trainer = Trainer(sess, env, agent, replay_buffer)
        
        if load:
            all_ckpt = tf.train.get_checkpoint_state(load_dir, 'checkpoint').all_model_checkpoint_paths
            saver.restore(sess, all_ckpt[-1])
        else:
            sess.run(tf.global_variables_initializer())
        
        random_return, goal_return = -200, 200
        lr_schedule = lambda x:3e-4*((goal_return-x)/(goal_return-random_return))
        eps_schedule = lambda x:((goal_return-x)/(goal_return-random_return))**2
        R, R_avg, L = trainer.train_online(max_episodes, batch_size, max_time=max_time,
                                        training_freq=1, repeat_action=repeat_action,
                                        gamma=gamma, eps_schedule=None,
                                        lr_schedule=None,
                                        render=render)
#        trainer.train_online(10, batch_size=1e10, max_time=max_time, render=True)
        
        env.close()
    
        if save:
            if os.path.isdir(save_dir): shutil.rmtree(save_dir)
            os.mkdir(save_dir)
            ckpt_path = os.path.join(save_dir, 'DDPG.ckpt')
            save_path = saver.save(sess, ckpt_path)
            print("\nSave Model %s\n" % save_path)

        return R, R_avg, L

if __name__ == '__main__':
    for v in [0]:
        tf.reset_default_graph()
#        clip = v
        R, R_avg, L = main()
        
        plt.figure()
        plt.grid()
        plt.title("")
        plt.plot(R, ".", label=str())
        plt.legend()
        plt.plot(R_avg)
#        plt.figure()
#        plt.plot(L)
