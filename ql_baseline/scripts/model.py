#scripts/model.py

import numpy as np
import random

class QTableAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.q_table = {}  # key = tuple(state), value = np.array of Q-values
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_q_values(self, state_key):
        """Returns Q-values for a state, initializing if unseen."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def select_action(self, state_key):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.get_q_values(state_key))

    def update(self, state_key, action, reward, next_state_key):
        """Q-learning update rule."""
        current_q = self.get_q_values(state_key)[action]
        next_max_q = np.max(self.get_q_values(next_state_key))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save the Q-table to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        """Load an existing Q-table."""
        import pickle
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

