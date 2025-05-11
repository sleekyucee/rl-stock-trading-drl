#agent

import torch
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from network import DQN, DuelingDQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, input_shape, num_actions, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.num_actions = num_actions
        
        #initialize networks
        ModelClass = DuelingDQN if config.get('dueling', False) else DQN
        self.policy_net = ModelClass(input_shape, num_actions).to(self.device)
        self.target_net = ModelClass(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #training setup
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        #training state
        self.epsilon = config['epsilon_start']
        self.learn_step_counter = 0
        self.current_loss = 0
        self.episode_reward = 0
        self.q_values = None

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.q_values = self.policy_net(state_tensor)
        return self.q_values.argmax().item()


    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.config['batch_size']:
            return None

        #sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config['batch_size']
        )
        
        #convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        #current Q values
        current_q = self.policy_net(states).gather(1, actions)

        #target Q values
        with torch.no_grad():
            if self.config.get('double_dqn', False):
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q

        #compute loss
        loss = torch.nn.MSELoss()(current_q, target_q)
        self.current_loss = loss.item()

        #optimize
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        #update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #epsilon decay
        self.epsilon = max(
            self.config['epsilon_min'],
            self.epsilon * self.config['epsilon_decay']
        )

        return self.current_loss

    def get_grad_norm(self):
        """Returns the L2 norm of gradients for monitoring"""
        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def get_metrics(self):
        """Returns current training metrics for logging"""
        return {
            'loss': self.current_loss,
            'reward': self.episode_reward,
            'epsilon': self.epsilon,
            'q_mean': self.q_values.mean().item() if self.q_values is not None else 0,
            'q_max': self.q_values.max().item() if self.q_values is not None else 0,
            'grad_norm': self.get_grad_norm()
        }

    def train_episode(self, states, prices, initial_cash):
        """Runs one full training episode"""
        self.episode_reward = 0
        cash, stock = initial_cash, 0
        
        for t in range(len(states) - 1):
            state = states[t]
            next_state = states[t + 1]
            price = prices[t]

            #select and execute action
            action = self.select_action(state)
            if action == 0 and cash >= price:
                stock += 1
                cash -= price
            elif action == 1 and stock > 0:
                cash += price
                stock -= 1

            #calculate reward
            next_price = prices[t + 1]
            portfolio = cash + stock * next_price
            current_value = cash + stock * price
            reward = portfolio - current_value
            self.episode_reward += reward

            #store experience
            self.store_transition(
                state, action, reward, next_state,
                (t == len(states) - 2)  # Done flag
            )

            #update network
            self.update()

        return self.episode_reward