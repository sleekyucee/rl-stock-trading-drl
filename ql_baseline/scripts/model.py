#model

import numpy as np
import random

class QTableAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay, policy_type="epsilon-greedy"):
        self.q_table = {}  #{(state): [q_values_for_actions]}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.policy_type = policy_type.lower()

    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        if self.policy_type == "greedy":
            return np.argmax(self.q_table[state])

        elif self.policy_type == "boltzmann":
            tau = max(self.epsilon, 0.01)  #use epsilon as temperature
            preferences = self.q_table[state] / tau
            probabilities = np.exp(preferences) / np.sum(np.exp(preferences))
            return np.random.choice(self.n_actions, p=probabilities)

        #default: epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        best_next_action = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next_action - self.q_table[state][action])

    def decay_epsilon(self):
        if self.policy_type == "epsilon-greedy":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)