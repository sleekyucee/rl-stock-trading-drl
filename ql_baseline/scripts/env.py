#scripts/env.py

import random

def reset_environment(stock_data_discrete, stock_data_continuous, window_size=30, initial_cash=10000):
    """
    Resets the environment for a new episode.

    Args:
        stock_data_discrete (pd.DataFrame): Discretized stock data.
        stock_data_continuous (pd.DataFrame): Original stock data with real prices.
        window_size (int): Number of time steps in the state window.
        initial_cash (float): Starting portfolio cash.

    Returns:
        state_discrete (pd.DataFrame): Initial discrete state.
        state_continuous (pd.DataFrame): Initial continuous state.
        current_index (int): Index at which the next step begins.
        portfolio (dict): Initial portfolio state.
    """
    start_index = random.randint(0, len(stock_data_discrete) - window_size - 1)

    state_discrete = stock_data_discrete.iloc[start_index : start_index + window_size]
    state_continuous = stock_data_continuous.iloc[start_index : start_index + window_size]

    portfolio = {
        'cash': initial_cash,
        'stock_held': 0,
        'total_value': initial_cash
    }

    return state_discrete, state_continuous, start_index + window_size, portfolio


def step_environment(action, market_data_row, portfolio):
    """
    Executes one environment step based on the selected action.

    Args:
        action (int): 0 = Buy, 1 = Sell, 2 = Hold
        market_data_row (pd.Series): Market data at current time step.
        portfolio (dict): Portfolio state to be updated.

    Returns:
        reward (float): Reward after taking the action.
        updated_portfolio (dict): Portfolio state after the action.
    """
    close_price = market_data_row['Close']

    if action == 0 and portfolio['stock_held'] == 0:
        portfolio['cash'] -= close_price
        portfolio['stock_held'] = 1

    elif action == 1 and portfolio['stock_held'] == 1:
        portfolio['cash'] += close_price
        portfolio['stock_held'] = 0

    # Calculate reward as the change in portfolio value
    portfolio_value = portfolio['cash'] + portfolio['stock_held'] * close_price
    reward = portfolio_value - portfolio['total_value']
    portfolio['total_value'] = portfolio_value

    return reward, portfolio

