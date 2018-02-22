# -*- coding: utf-8 -*-

import numpy as np
import time

INF = float('inf')

class Trainer():
    
    def __init__(self, sess, env, agent, replay_buffer):
        self.sess = sess
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        
    def train_online(self, max_episodes, batch_size, max_episode_len=INF, max_steps=INF,
              max_time=INF, training_freq=1, printing_freq=1, render=False):
        # Initialize agent (target network)
        self.agent.init_step() # Useful ???
        
        t_start = int(time.time())
        t_current = 0
        ep_rewards, avg_rewards = [], []
        step = 0
        episode = 0
        while episode<max_episodes and step<max_steps and t_current<max_time:
            # Start a new episode
            episode += 1
            s = self.env.reset()
           
            self.agent.restart_step() # Noise decay, ...
                      
            ep_step = 0
            ep_reward = 0
            done = False
            
            while ep_step<max_episode_len and not done:
                step += 1
                ep_step += 1
                
                if render:
                    self.env.render()
                    
                # Get action from agent and take it
                s_noisy = np.random.normal(s, 0.01)
                a = self.agent.select_action(s)
                s2, r, done, info = self.env.step(a)
                ep_reward += r
                # Save experience in buffer
                self.replay_buffer.add( (s, a, r, done, s2) )
                s = s2
                
                # Train the agent
                if step%training_freq==0:
                    self.agent.training_step(batch_size, self.replay_buffer)
    
            t_current = int(time.time()) - t_start
  
            # Bookkeeping (only word with 3 consecutive double letters)
            ep_rewards.append(ep_reward)
            avg_reward = np.mean(ep_rewards[-100:])
            avg_rewards.append(avg_reward)
            if episode%printing_freq==0:
                minutes, seconds = divmod(t_current, 60)
                hours, minutes = divmod(minutes, 60)  
                time_string = (hours>0)*(str(hours)+'h') + (minutes>0)*(str(minutes)+'m') + str(seconds)+'s'
                print('| {:s}:{:d} | Ep: {:d} | R: {:d} | AVG: {:.2f} |'.format(
                        time_string, step, episode,
                        int(ep_reward), avg_reward))
         
        return ep_rewards, avg_rewards

