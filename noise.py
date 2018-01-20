# -*- coding: utf-8 -*-
import numpy as np

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class ParameterNoise():
    def __init__(self, initial_std=0.1, target_policy_std=0.2, adaptation_coef=1.01):
        self.std = initial_std
        self.target_policy_std = target_policy_std
        self.adaptation_coef = adaptation_coef

    def adapt(self, distance):
        if distance > self.target_policy_std:
            # Decrease stddev.
            self.std /= self.adaptation_coef
        else:
            # Increase stddev.
            self.std *= self.adaptation_coef
            
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)