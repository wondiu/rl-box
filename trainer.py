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
              max_time=INF, gamma=0.99, lr_schedule=None, eps_schedule=None, training_freq=1, repeat_action=1, printing_freq=1, render=False):
        if lr_schedule==None: lr_schedule=lambda x:3e-4
        if eps_schedule==None: eps_schedule=lambda x:0.97**episode
        self.agent.init_step() # Useful ???
        
        t_start = int(time.time())
        t_current = 0
        ep_rewards, avg_rewards = [], []
        ep_reward, avg_reward = 0, 0
        losses = []
        step = 0
        episode = 0
        while episode<max_episodes and step<max_steps and t_current<max_time:
            # Start a new episode
            episode += 1
            s = self.env.reset()
           
            self.agent.epsilon_update(eps_schedule(ep_reward))
                      
            ep_step = 0
            ep_reward = 0
            done = False
            
            while ep_step<max_episode_len and not done:
                step += 1
                ep_step += 1
                
                if render:# or episode%20==0:
                    self.env.render()
                    
                # Get action from agent and take it
#                s_noisy = np.random.normal(s, 0.01)
                if (step-1)%repeat_action==0:
                    a = self.agent.select_action(s)
                s2, r, done, info = self.env.step(a)
                ep_reward += r
                # Save experience in buffer
                self.replay_buffer.add( (s, a, r, done, s2) )
                s = s2
                
                if step%100:
                        self.agent.update_target_net(1)
                        
                # Train the agent
                if step%training_freq==0:
                    tolerance=10
                    loss=tolerance+1
                    while loss!=None and loss>tolerance:
                        tolerance+=1
                        loss = self.agent.training_step(batch_size, self.replay_buffer,
                                                 gamma=gamma, lr=lr_schedule(avg_reward))
                    if loss is not None:
                        losses.append(loss)
                    
            t_current = int(time.time()) - t_start
  
            # Bookkeeping (only word with 3 consecutive double letters)
            ep_rewards.append(ep_reward)
            avg_reward = np.mean(ep_rewards[-100:])
            avg_rewards.append(avg_reward)
            ep_loss = np.mean(losses[-100:])
            if episode%printing_freq==0:
                minutes, seconds = divmod(t_current, 60)
                hours, minutes = divmod(minutes, 60)  
                time_string = (hours>0)*(str(hours)+'h') + (minutes>0)*(str(minutes)+'m') + str(seconds)+'s'
                print('| {:s}:{:d} | Ep: {:d} | R: {:d} | AvgR: {:.2f} | LR: {:.1e} | Eps: {:.1g} | Loss: {:.3g}'.format(
                        time_string, step, episode,
                        int(ep_reward), avg_reward, lr_schedule(avg_reward), self.agent.epsilon,
                        ep_loss))
            if avg_reward>=199:
                break
         
        return ep_rewards, avg_rewards, losses

