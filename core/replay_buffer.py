"""Finite replay buffer."""

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, policy, value, performance):
        self.buffer.append((state, policy, value, performance))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        states, policies, values, performances = zip(*batch)
        return list(states), list(policies), list(values), list(performances)

    def __len__(self):
        return len(self.buffer)