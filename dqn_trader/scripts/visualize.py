#visualize

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_total_reward(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

def plot_moving_average(rewards, window=10):
    avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(avg)
    plt.title(f"Moving Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True)
    plt.show()

def plot_final_portfolio(portfolio_values):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title("Final Portfolio Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.show()

def plot_portfolio_over_time(portfolio_values):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step (Day)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.show()

def visualize(config_name, base_path="dqn_trader"):
    rewards_path = os.path.join(base_path, f"rewards_{config_name}.npy")
    portfolio_path = os.path.join(base_path, f"portfolio_{config_name}.npy")

    try:
        rewards = np.load(rewards_path)
        portfolio = np.load(portfolio_path)

        plot_total_reward(rewards)
        plot_moving_average(rewards)
        plot_final_portfolio(portfolio)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Ensure the files exist and the config name is correct.")
