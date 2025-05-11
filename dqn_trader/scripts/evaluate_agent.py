#evaluate_agent

import os
import random
import numpy as np
import torch

from dataset import load_data
from utils import load_config
from agent import DQNAgent

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_deterministic(seed):
    """Ensure reproducible behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def benchmark_performance(prices, initial_cash):
    return (prices[-1] / prices[0]) * initial_cash

def calculate_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return {
        'sharpe': np.mean(returns) / (np.std(returns) + 1e-9),
        'sortino': np.mean(returns) / (np.std(returns[returns < 0]) + 1e-9),
        'max_drawdown': (np.max(portfolio_values) - np.min(portfolio_values)) / np.max(portfolio_values)
    }

def run_evaluation(config_path, model_path):
    config = load_config(config_path)
    model_name = config['experiment_settings']['experiment_name']

    #set seed for reproducibility
    set_deterministic(config['train_settings'].get('seed', 42))

    #load test data
    states, prices, _, _ = load_data(
        csv_path=config['data_settings']['test_csv_path'],
        window_size=config['data_settings']['window_size']
    )

    #initialize agent
    agent = DQNAgent(
        input_shape=states.shape[1:],
        num_actions=3,
        config=config['train_settings']
    )
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.to(device)
    agent.policy_net.eval()

    #evaluate
    portfolio_history = []
    cash = config['env_settings']['initial_cash']
    stock = 0

    for t in range(len(states) - 1):
        state = states[t]
        action = agent.select_action(state, eval_mode=True)
        price = prices[t]

        if action == 0 and cash >= price:
            stock += 1
            cash -= price
        elif action == 1 and stock > 0:
            cash += price
            stock -= 1

        portfolio_history.append(cash + stock * prices[t + 1])

    #compute metrics
    metrics = calculate_metrics(portfolio_history)
    bh_value = benchmark_performance(prices, config['env_settings']['initial_cash'])

    #save results
    os.makedirs("eval_results", exist_ok=True)
    np.savez(f"eval_results/{model_name}_test.npz",
             portfolio=np.array(portfolio_history),
             prices=prices)

    #print report
    print("\n=== Final Evaluation ===")
    print(f"Strategy: ${portfolio_history[-1]:.2f}")
    print(f"Buy & Hold: ${bh_value:.2f}")
    print(f"Outperformance: ${portfolio_history[-1] - bh_value:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"\nResults saved to: eval_results/{model_name}_test.npz")

if __name__ == "__main__":
    import sys
    run_evaluation(sys.argv[1], sys.argv[2])