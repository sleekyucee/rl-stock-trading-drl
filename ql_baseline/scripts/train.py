#train

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

#add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from scripts.utils import load_config, set_seed
from scripts.dataset import load_stock_data, fit_discretizer, apply_discretizer
from scripts.env import reset_environment, step_environment
from scripts.model import QTableAgent
from scripts.logger import Logger

def main(config_path):
    config = load_config(config_path)

    #set seed
    set_seed(config['train_settings']['seed'])

    #load and preprocess data
    df_continuous = load_stock_data(config['data_settings']['csv_path'])

    #fit and apply discretizer on training data
    discretizer = fit_discretizer(df_continuous, n_bins=config['data_settings']['n_bins'])
    df_discrete = apply_discretizer(df_continuous, discretizer)

    #save the discretizer
    discretizer_path = os.path.join("models", f"discretizer_{config['experiment_settings']['experiment_name']}.pkl")
    os.makedirs(os.path.dirname(discretizer_path), exist_ok=True)
    with open(discretizer_path, "wb") as f:
        pickle.dump(discretizer, f)

    #initialize WandB logger
    logger = Logger(
        experiment_name=config['experiment_settings']['experiment_name'],
        project=config['experiment_settings']['project']
    ) if config['experiment_settings']['use_wandb'] else None

    #determine policy type
    policy_type = config['experiment_settings'].get("policy_type", "epsilon-greedy")

    #initialize agent
    agent = QTableAgent(
        n_actions=3,
        alpha=config['train_settings']['alpha'],
        gamma=config['train_settings']['gamma'],
        epsilon=config['train_settings']['epsilon'],
        epsilon_min=config['train_settings']['epsilon_min'],
        epsilon_decay=config['train_settings']['epsilon_decay'],
        policy_type=policy_type
    )

    rewards_per_episode = []
    final_portfolio_values = []

    #training loop
    for episode in tqdm(range(config['train_settings']['n_episodes']), desc="Training episodes"):
        state_disc, state_cont, idx, portfolio = reset_environment(
            df_discrete, df_continuous,
            window_size=config['data_settings']['window_size'],
            initial_cash=config['env_settings']['initial_cash']
        )

        state_key = tuple(state_disc.values.flatten())
        total_reward = 0
        done = False

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

    #save model and results
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    filename = f"q_table_{config['experiment_settings']['experiment_name']}.pkl"
    agent.save(os.path.join(model_dir, filename))

    np.save(f"rewards_{config['experiment_settings']['experiment_name']}.npy", rewards_per_episode)
    np.save(f"portfolio_{config['experiment_settings']['experiment_name']}.npy", final_portfolio_values)

    print(f"Training complete for {config['experiment_settings']['experiment_name']}")

    if logger:
        logger.save_file(filename)
        logger.finish()

#support CLI execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)

    main(sys.argv[1])