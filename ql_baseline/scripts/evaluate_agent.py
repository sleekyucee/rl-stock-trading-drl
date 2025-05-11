#evaluate

import os
import pickle
#evaluate_agent

import numpy as np
import pandas as pd
from utils import load_config, set_seed
from dataset import load_stock_data, apply_discretizer
from env import reset_environment, step_environment
from model import QTableAgent
from visualize import plot_portfolio_over_time, plot_best_action_distribution
from scipy.spatial import distance

def find_closest_state_key(state_key, q_table_keys):
    """Finds the closest known Q-table key using Euclidean distance."""
    state_array = np.array(state_key)
    known_keys = np.array(q_table_keys)
    dists = np.linalg.norm(known_keys - state_array, axis=1)
    return tuple(known_keys[np.argmin(dists)])

def evaluate(config_path, model_path):
    # Load configuration
    config = load_config(config_path)
    print(f"Loading config from: {config_path}")
    print(f"Loading model from: {model_path}")
    
    #set seed
    set_seed(config['train_settings']['seed'])

    #load data
    test_csv = config['data_settings'].get('test_csv_path', config['data_settings']['csv_path'])
    df_continuous = load_stock_data(test_csv)

    #load discretizer
    with open(config["eval_settings"]["discretizer_path"], "rb") as f:
        discretizer = pickle.load(f)

    df_discrete = apply_discretizer(df_continuous, discretizer)

    # Load agent
    agent = QTableAgent(
        n_actions=3,
        alpha=config['train_settings']['alpha'],
        gamma=config['train_settings']['gamma'],
        epsilon=0.0,  # greedy evaluation
        epsilon_min=0.0,
        epsilon_decay=1.0
    )
    agent.load(model_path)

    #environment setup
    state_disc, state_cont, idx, portfolio = reset_environment(
        df_discrete, df_continuous,
        window_size=config['data_settings']['window_size'],
        initial_cash=config['env_settings']['initial_cash']
    )

    state_key = tuple(map(lambda x: int(round(x)), state_disc.values.flatten()))
    done = False
    portfolio_values = [portfolio['total_value']]
    rewards = []
    total_steps = 0
    matched_states = 0
    q_keys_array = np.array(list(agent.q_table.keys()))

    while not done:
        total_steps += 1

        if state_key in agent.q_table:
            matched_states += 1
            action = agent.select_action(state_key)
        else:
            print(f"[WARNING] Q-table miss at step {total_steps}, selecting nearest state.")
            if len(q_keys_array) > 0:
                fallback_key = find_closest_state_key(state_key, q_keys_array)
                action = np.argmax(agent.q_table[fallback_key])
            else:
                action = np.random.choice([0, 1, 2])  # true fallback

        # Step forward
        next_disc_row = df_discrete.iloc[idx]
        next_cont_row = df_continuous.iloc[idx]
        reward, portfolio = step_environment(action, next_cont_row, portfolio)
        rewards.append(reward)
        portfolio_values.append(portfolio['total_value'])

        next_state_disc = pd.concat([state_disc.iloc[1:], next_disc_row.to_frame().T])
        next_state_key = tuple(map(lambda x: int(round(x)), next_state_disc.values.flatten()))

        state_disc = next_state_disc
        state_key = next_state_key
        idx += 1

        if idx >= len(df_discrete) - 1:
            done = True

    #final stats
    final_value = portfolio['total_value']
    initial_value = portfolio_values[0]
    total_return = final_value - initial_value
    return_pct = (total_return / initial_value) * 100
    match_pct = (matched_states / total_steps) * 100

    print(f"\nFinal Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: ${total_return:.2f} ({return_pct:.2f}%)")
    print(f"Q-table match: {matched_states}/{total_steps} steps ({match_pct:.2f}%)")

    #visuals
    experiment_name = config['experiment_settings']['experiment_name']
    plot_portfolio_over_time(portfolio_values, experiment_name)
    plot_best_action_distribution(agent.q_table, experiment_name)

    #save results
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)
    np.save(os.path.join(eval_dir, f"portfolio_eval_{experiment_name}.npy"), portfolio_values)
    np.save(os.path.join(eval_dir, f"rewards_eval_{experiment_name}.npy"), rewards)

    print(f"Evaluation results saved to: {eval_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate_agent.py <config_path> <model_path>")
    else:
        evaluate(sys.argv[1], sys.argv[2])
