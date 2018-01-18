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
