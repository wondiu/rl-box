# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
import random

class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen = int(max_size))

    def add(self, datum):
        self.buffer.append(datum)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, min(self.size(), batch_size))
        return [np.array([_[i] for _ in batch]) for i in range(len(batch[0]))]

    def clear(self):
        self.buffer.clear()


class SplitBuffer():
    def __init__(self, max_size, n_buffers=2, thresholds=[0]):
        self.max_size = max_size
        self.n_buffers = n_buffers
        self.thresholds = thresholds
        self.buffers = [deque(maxlen = int(max_size//n_buffers)) for i in range(n_buffers)]

    def add(self, datum, ep_reward):
        i = 0
        while i<self.n_buffers-1 and ep_reward>self.thresholds[i]:
            i+=1
        self.buffers[i].append(datum)

    def size(self):
        return sum([len(buffer) for buffer in self.buffers])

    def sample_batch(self, batch_size):
        batch = []
        for i in range(self.n_buffers-1,-1,-1):
            buffer_batch_size = min(len(self.buffers[i]), batch_size//(i+1))
            batch = batch + random.sample(self.buffers[i], buffer_batch_size)
            batch_size = batch_size - buffer_batch_size
        
        return [np.array([_[i] for _ in batch]) for i in range(len(batch[0]))]

    def clear(self):
        for buffer in self.buffers:
            buffer.clear()
