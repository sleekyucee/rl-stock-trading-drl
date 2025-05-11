#network

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        input_shape: tuple (window_size, num_features)
        num_actions: number of discrete actions (e.g., 3 for Buy, Sell, Hold)
        """
        super(DQN, self).__init__()
        self.window_size, self.num_features = input_shape

        self.fc1 = nn.Linear(self.window_size * self.num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  #flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Dueling architecture separates value and advantage streams
        """
        super(DuelingDQN, self).__init__()
        self.window_size, self.num_features = input_shape

        self.feature_layer = nn.Sequential(
            nn.Linear(self.window_size * self.num_features, 256),
            nn.ReLU()
        )

        #value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        #advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  #flatten
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        #combine using dueling formula: Q = V + (A - mean(A))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
