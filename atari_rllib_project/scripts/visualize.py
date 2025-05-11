#visualize

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_metric(file_path, title="Reward Trend", ylabel="Reward"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    values = np.load(file_path)
    plt.figure(figsize=(10, 4))
    plt.plot(values, label=ylabel)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .npy reward/loss/metric file")
    parser.add_argument("--title", default="Metric Trend", help="Plot title")
    parser.add_argument("--ylabel", default="Value", help="Y-axis label")
    args = parser.parse_args()

    plot_metric(args.file, title=args.title, ylabel=args.ylabel)
