#scripts/visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_total_reward(rewards):
    plt.figure(figsize=(10,6))
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

def plot_moving_average(rewards, window=10):
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,6))
    plt.plot(moving_avg)
    plt.title(f"Moving Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True)
    plt.show()

def plot_final_portfolio(portfolio_values):
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_values)
    plt.title("Final Portfolio Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.show()

def plot_best_action_distribution(q_table):
    action_counts = {0: 0, 1: 0, 2: 0}
    for q_values in q_table.values():
        best_action = np.argmax(q_values)
        action_counts[best_action] += 1

    actions = ['Buy', 'Sell', 'Hold']
    counts = [action_counts[0], action_counts[1], action_counts[2]]

    plt.figure(figsize=(8, 5))
    plt.bar(actions, counts)
    plt.title("Distribution of Best Actions in Learned Q-Table")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.show()

def plot_portfolio_over_time(portfolio_values):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time (During Testing)")
    plt.xlabel("Time Step (Day)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.show()

