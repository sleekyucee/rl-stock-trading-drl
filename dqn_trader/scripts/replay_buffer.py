#replay_buffer

import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return (
            np.array(batch.state),
            np.array(batch.action),
            np.array(batch.reward),
            np.array(batch.next_state),
            np.array(batch.done),
        )

    def __len__(self):
        return len(self.buffer)
