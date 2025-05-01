#scripts/evaluate_agent.py

import argparse
import numpy as np
import pandas as pd

from scripts.utils import load_config, set_seed
from scripts.dataset import load_stock_data, discretize_features
from scripts.env import reset_environment, step_environment
from scripts.model import QTableAgent
from scripts.visualize import plot_portfolio_over_time, plot_best_action_distribution

def evaluate(config, model_path):
    #set seed
    set_seed(config['train_settings']['seed'])

    #load data
    df_continuous = load_stock_data(config['data_settings']['csv_path'])
    df_discrete, _ = discretize_features(df_continuous, n_bins=config['data_settings']['n_bins'])

    #load agent
    agent = QTableAgent(
        n_actions=3,
        alpha=config['train_settings']['alpha'],
        gamma=config['train_settings']['gamma'],
        epsilon=0.0,  #no exploration during evaluation
        epsilon_min=0.0,
        epsilon_decay=1.0
    )
    agent.load(model_path)

    #reset environment
    state_disc, state_cont, idx, portfolio = reset_environment(
        df_discrete, df_continuous,
        window_size=config['data_settings']['window_size'],
        initial_cash=config['env_settings']['initial_cash']
    )

    state_key = tuple(state_disc.values.flatten())
    done = False

    portfolio_values = [portfolio['total_value']]

    while not done:
        action = agent.select_action(state_key)

        next_disc_row = df_discrete.iloc[idx]
        next_cont_row = df_continuous.iloc[idx]

        reward, portfolio = step_environment(action, next_cont_row, portfolio)
        portfolio_values.append(portfolio['total_value'])

        next_state_disc = pd.concat([state_disc.iloc[1:], next_disc_row.to_frame().T])
        next_state_key = tuple(next_state_disc.values.flatten())

        state_disc = next_state_disc
        state_key = next_state_key
        idx += 1

        if idx >= len(df_discrete) - 1:
            done = True

    #plot results
    print(f"Final Portfolio Value: ${portfolio['total_value']:.2f}")
    plot_portfolio_over_time(portfolio_values)
    plot_best_action_distribution(agent.q_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Q-table")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.model)

