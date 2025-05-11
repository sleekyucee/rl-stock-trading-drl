#visualize
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_total_reward(rewards, config_name):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_total_reward.png"))
    plt.close()

def plot_moving_average(rewards, config_name, window=10):
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg)
    plt.title(f"Moving Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_moving_avg.png"))
    plt.close()

def plot_final_portfolio(portfolio_values, config_name):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title("Final Portfolio Value per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_portfolio.png"))
    plt.close()

def plot_best_action_distribution(q_table, config_name):
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
    plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_action_distribution.png"))
    plt.close()

def plot_portfolio_over_time(portfolio_values, config_name):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time (During Testing)")
    plt.xlabel("Time Step (Day)")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_portfolio_time.png"))
    plt.close()

def visualize(config_name):
    rewards_path = f"rewards_{config_name}.npy"
    portfolio_path = f"portfolio_{config_name}.npy"
    q_table_path = os.path.join("models", f"q_table_{config_name}.pkl")

    try:
        rewards = np.load(rewards_path)
        portfolio = np.load(portfolio_path)
        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)

        plot_total_reward(rewards, config_name)
        plot_moving_average(rewards, config_name)
        plot_final_portfolio(portfolio, config_name)
        plot_best_action_distribution(q_table, config_name)
        plot_portfolio_over_time(portfolio, config_name)

        print(f"Saved all plots for {config_name} in '{PLOT_DIR}/'")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Ensure the file paths are correct and the config name matches your saved outputs.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <config_name>")
    else:
        visualize(sys.argv[1])
