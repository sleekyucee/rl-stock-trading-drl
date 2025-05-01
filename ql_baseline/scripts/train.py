#scripts/train.py

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.utils import load_config, set_seed
from scripts.dataset import load_stock_data, discretize_features
from scripts.env import reset_environment, step_environment
from scripts.model import QTableAgent
from scripts.logger import Logger

def main(config):
    # Set seed
    set_seed(config['train_settings']['seed'])

    # Load and preprocess data
    df_continuous = load_stock_data(config['data_settings']['csv_path'])
    df_discrete, _ = discretize_features(df_continuous, n_bins=config['data_settings']['n_bins'])

    # WandB logger
    logger = None
    if config['experiment_settings']['use_wandb']:
        logger = Logger(
            experiment_name=config['experiment_settings']['experiment_name'],
            project=config['experiment_settings']['project']
        )

    # Agent
    agent = QTableAgent(
        n_actions=3,
        alpha=config['train_settings']['alpha'],
        gamma=config['train_settings']['gamma'],
        epsilon=config['train_settings']['epsilon'],
        epsilon_min=config['train_settings']['epsilon_min'],
        epsilon_decay=config['train_settings']['epsilon_decay']
    )

    # Logs
    rewards_per_episode = []
    final_portfolio_values = []

    for episode in tqdm(range(config['train_settings']['n_episodes']), desc="Training episodes"):
        state_disc, state_cont, idx, portfolio = reset_environment(
            df_discrete, df_continuous,
            window_size=config['data_settings']['window_size'],
            initial_cash=config['env_settings']['initial_cash']
        )

        state_key = tuple(state_disc.values.flatten())
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state_key)

            next_disc_row = df_discrete.iloc[idx]
            next_cont_row = df_continuous.iloc[idx]
            reward, portfolio = step_environment(action, next_cont_row, portfolio)
            total_reward += reward

            next_state_disc = pd.concat([state_disc.iloc[1:], next_disc_row.to_frame().T])
            next_state_key = tuple(next_state_disc.values.flatten())

            agent.update(state_key, action, reward, next_state_key)

            state_disc = next_state_disc
            state_key = next_state_key
            idx += 1

            if idx >= len(df_discrete) - 1:
                done = True

        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        final_portfolio_values.append(portfolio['total_value'])

        if logger:
            logger.log({
                "reward": total_reward,
                "portfolio_value": portfolio['total_value'],
                "epsilon": agent.epsilon
            })

    # Save outputs
    os.makedirs("qlearning_baseline/models", exist_ok=True)
    model_path = f"qlearning_baseline/models/q_table_{config['experiment_settings']['experiment_name']}.pkl"
    agent.save(model_path)

    if logger:
        logger.save_file(model_path)
        logger.finish()

    np.save(f"qlearning_baseline/rewards_{config['experiment_settings']['experiment_name']}.npy", rewards_per_episode)
    np.save(f"qlearning_baseline/portfolio_{config['experiment_settings']['experiment_name']}.npy", final_portfolio_values)

    print(f"Training complete for {config['experiment_settings']['experiment_name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

