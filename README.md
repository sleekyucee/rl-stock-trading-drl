#Deep Reinforcement Learning for Stock Trading

This project explores reinforcement learning techniques for algorithmic stock trading using historical TSLA stock data. The implementation includes both a tabular Q-learning agent and advanced deep reinforcement learning agents (DQN and Double DQN), trained on a custom environment built from 15 years of real stock market data.

Project Overview

- Basic Task: A tabular Q-learning agent learns trading decisions (Buy, Sell, Hold) by interacting with a simplified stock trading environment. The state space is discretized using KBinsDiscretizer, and the agent updates a Q-table using an ε-greedy policy.

- Advanced Task: A Deep Q-Network (DQN) and a Double DQN agent are implemented using PyTorch. These agents operate on a high-dimensional state space derived from a 30-day rolling window of 13 technical indicators. Experience replay and target networks are used to stabilize learning.

Dataset

The dataset consists of approximately 15 years of daily TSLA stock price data, enriched with the following technical indicators:

- Price: Open, High, Low, Close

- Volume

- Technical Indicators: Percentage change, SMA30, SMA100, RSI14, MACD, Bollinger Bands, Volatility

Tasks Summary

- Basic Q-Learning Agent

- Custom trading environment with 30-day rolling window

- State discretization using scikit-learn's KBinsDiscretizer

- Q-table learning with epsilon-greedy exploration strategy

- Performance evaluation through reward and portfolio growth plots

Advanced DQN & Double DQN Agents

- PyTorch-based deep reinforcement learning

- State vector flattened to 390 input dimensions (30 x 13 features)

- Experience replay buffer and target network used

- Double DQN implemented to reduce overestimation bias

- Training performed on Hyperion GPU cluster

Setup

To install required packages, run:

pip install -r requirements.txt

