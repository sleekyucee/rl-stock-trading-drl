#train

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

#project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import load_data
from scripts.agent import DQNAgent
from scripts.logger import Logger
from scripts.utils import load_config, set_seed

def evaluate_agent(agent, states, prices, initial_cash):
    """Evaluation with risk metrics"""
    agent.epsilon = 0.0  # Greedy policy
    portfolio_values = []
    cash, stock = initial_cash, 0
    
    for t in range(len(states) - 1):
        state = states[t]
        action = agent.select_action(state)
        price = prices[t]
        
        # Position sizing
        if action == 0 and cash >= price:
            stock += 1
            cash -= price
        elif action == 1 and stock > 0:
            cash += price
            stock -= 1
        
        portfolio_values.append(cash + stock * prices[t + 1])
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    metrics = {
        'final_value': portfolio_values[-1],
        'sharpe': np.mean(returns) / (np.std(returns) + 1e-9),
        'max_drawdown': (np.max(portfolio_values) - np.min(portfolio_values)) / np.max(portfolio_values),
        'portfolio_history': portfolio_values  #added for saving
    }
    return metrics

def main():
    config = load_config(sys.argv[1])
    set_seed(config['train_settings']['seed'])
    model_name = config['experiment_settings']['experiment_name']
    
    #load data
    states, prices, scaler, _ = load_data(
        csv_path=config['data_settings']['csv_path'],
        window_size=config['data_settings']['window_size']
    )
    
    #initialize
    agent = DQNAgent(
        input_shape=states.shape[1:],
        num_actions=3,
        config=config['train_settings']
    )
    logger = Logger(
        experiment_name=model_name,
        project=config['experiment_settings']['project']
    ) if config['experiment_settings']['use_wandb'] else None

    #training records
    train_rewards = []
    train_portfolios = []
    val_rewards = []
    val_portfolios = []
    losses = []

    #training loop
    for episode in tqdm(range(config['train_settings']['n_episodes']), desc=f"Training {model_name}"):
        #train episode
        episode_reward = agent.train_episode(states, prices, config['env_settings']['initial_cash'])
        train_rewards.append(episode_reward)
        losses.append(agent.current_loss)

        #logging
        if logger and episode % 10 == 0:
            logger.log(agent.get_metrics())

        #validation
        if (episode + 1) % config['train_settings'].get('eval_freq', 50) == 0:
            val_metrics = evaluate_agent(agent, *load_data(
                csv_path=config['data_settings']['valid_csv_path'],
                window_size=config['data_settings']['window_size'],
                scaler=scaler
            )[:2], config['env_settings']['initial_cash'])
            
            val_rewards.append(np.mean(val_metrics['portfolio_history']))
            val_portfolios.append(val_metrics['final_value'])
            
            if logger:
                logger.log({f'val_{k}': v for k, v in val_metrics.items()})

    #save everything
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    #save model
    torch.save(agent.policy_net.state_dict(), f"models/policy_net_{model_name}.pt")
    
    #save training history
    np.savez(f"results/{model_name}_history.npz",
             train_rewards=np.array(train_rewards),
             train_portfolios=np.array(train_portfolios),
             val_rewards=np.array(val_rewards),
             val_portfolios=np.array(val_portfolios),
             losses=np.array(losses))
    
    print(f"\nTraining complete for {model_name}")
    print(f"Model saved to: models/policy_net_{model_name}.pt")

if __name__ == "__main__":
    main()