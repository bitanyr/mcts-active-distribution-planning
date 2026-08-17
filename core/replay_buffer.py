# core / replay buffer
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    
    def push(self, state, pi, value, perf):
        
        self.buffer.append((state, pi, value, perf))

    def sample(self, batch_size):
        
        batch = random.sample(self.buffer, batch_size)
        
        states, pis, values, perfs = zip(*batch)
        return list(states), list(pis), list(values), list(perfs)

    def __len__(self):
        
        return len(self.buffer)