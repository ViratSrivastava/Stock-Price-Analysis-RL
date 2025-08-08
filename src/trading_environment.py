# trading_environment.py
"""
Custom trading environment for reinforcement learning
Simulates stock trading with realistic constraints and portfolio management
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any

class StockTradingEnvironment(gym.Env):
    """Custom gym environment for stock trading with DQN"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 transaction_cost: float = 0.001, max_shares: int = 1000):
        super(StockTradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares
        
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: all features + portfolio info
        feature_dim = len(self.data.columns)
        portfolio_dim = 5  # balance, shares, net_worth, position_value, portfolio_return
        total_dim = feature_dim + portfolio_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        # Portfolio state
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_history = []
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        # Reset logging flags for new episode
        if hasattr(self, '_logged_fallback_this_episode'):
            delattr(self, '_logged_fallback_this_episode')
        if hasattr(self, '_logged_rolling_this_episode'):
            delattr(self, '_logged_rolling_this_episode')
        if hasattr(self, '_logged_global_this_episode'):
            delattr(self, '_logged_global_this_episode')
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.trade_history = []
        self.portfolio_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
            
        # Market features
        market_obs = self.data.iloc[self.current_step].values
        
        # Portfolio information (normalized)
        current_price = self.data.iloc[self.current_step]['close']
        position_value = self.shares_held * current_price
        
        portfolio_obs = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held / self.max_shares,   # Normalized shares
            self.net_worth / self.initial_balance, # Normalized net worth
            position_value / self.initial_balance, # Normalized position value
            (self.net_worth - self.initial_balance) / self.initial_balance  # Portfolio return
        ])
        
        observation = np.concatenate([market_obs, portfolio_obs]).astype(np.float32)
        
        # Handle any infinite or NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
            
        # Replace direct price access with the validated helper method
        current_price = self._get_current_price()
        previous_net_worth = self.net_worth
        
        # Execute action
        reward = 0
        
        if action == 1:  # Buy
            available_balance = self.balance * (1 - self.transaction_cost)
            shares_to_buy, total_cost = self._execute_buy_action(current_price, available_balance)
            if shares_to_buy > 0:
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': total_cost
                })
                reward = 0.01
        elif action == 2:  # Sell
            if self.shares_held > 0:
                reward = self._handle_sell_action(current_price)
        
        # Update portfolio value
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Track portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'shares': self.shares_held,
            'net_worth': self.net_worth,
            'action': action
        })
        
        # Calculate reward based on portfolio performance
        portfolio_return = (self.net_worth - previous_net_worth) / previous_net_worth
        reward += portfolio_return * 10  # Scale the reward
        
        # Add risk penalty for excessive trading
        if action != 0:  # Not holding
            reward -= 0.01  # Small penalty for trading
            
        # Bonus for beating market
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            reward += 0.05
            
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'portfolio_return': (self.net_worth - self.initial_balance) / self.initial_balance
        }
        
        return self._get_observation(), reward, done, info

    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return immediate reward"""
        if action == 1:  # Buy
            return self._handle_buy_action(current_price)
        elif action == 2:  # Sell
            return self._handle_sell_action(current_price)
        return 0.0  # Hold action
    
    def _handle_buy_action(self, current_price: float) -> float:
        """Handle buy action execution"""
        available_balance = self.balance * (1 - self.transaction_cost)
        shares_to_buy, total_cost = self._execute_buy_action(current_price, available_balance)
        
        if not shares_to_buy:
            return 0.0
            
        self.balance -= total_cost
        self.shares_held += shares_to_buy
        self.total_trades += 1
        
        self.trade_history.append({
            'step': self.current_step,
            'action': 'BUY',
            'shares': shares_to_buy,
            'price': current_price,
            'cost': total_cost
        })
        
        return 0.01  # Buy reward
        
    def _handle_sell_action(self, current_price: float) -> float:
        """Handle sell action execution"""
        if self.shares_held == 0:
            return 0.0
            
        total_revenue = self.shares_held * current_price * (1 - self.transaction_cost)
        reward = self._calculate_sell_reward(current_price)
        
        self.balance += total_revenue
        self.trade_history.append({
            'step': self.current_step,
            'action': 'SELL',
            'shares': self.shares_held,
            'price': current_price,
            'revenue': total_revenue
        })
        
        self.shares_held = 0
        self.total_trades += 1
        return reward
        
    def _calculate_sell_reward(self, current_price: float) -> float:
        """Calculate reward for sell action"""
        if not self.trade_history:
            return 0.0
            
        last_buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        if not last_buy_trades:
            return 0.0
            
        avg_buy_price = np.mean([t['price'] for t in last_buy_trades])
        if current_price > avg_buy_price:
            self.winning_trades += 1
            return 0.02  # Profitable trade reward
        return -0.01  # Loss penalty

    def _execute_buy_action(self, current_price: float, available_balance: float) -> Tuple[int, float]:
        """Execute buy action with proper error handling"""
        
        # Fix: Check for zero or invalid price
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            self.logger.warning(f"Invalid price detected: {current_price}, skipping buy action")
            return 0, 0.0
        
        # Calculate affordable shares with safety check
        try:
            max_affordable_shares = int(available_balance // current_price)
            
            # Additional safety: ensure positive values
            if max_affordable_shares <= 0:
                return 0, 0.0
                
            # Limit position size (risk management)
            max_position_shares = int(self.initial_balance * 0.1 // current_price)  # Max 10% position
            shares_to_buy = min(max_affordable_shares, max_position_shares)
            
            if shares_to_buy > 0:
                total_cost = shares_to_buy * current_price
                transaction_cost = total_cost * self.transaction_cost
                return shares_to_buy, total_cost + transaction_cost
            else:
                return 0, 0.0
                
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            self.logger.error(f"Error in buy calculation: {e}")
            return 0, 0.0

    def _get_current_price(self) -> float:
        """Get current price with validation - IMPROVED VERSION"""
        try:
            price = self.data.iloc[self.current_step]['close']
            
            # Validate price
            if pd.isna(price) or price <= 0:
                # Try recent valid prices - search back up to 5 steps
                for i in range(1, 6):
                    prev_step = self.current_step - i
                    if prev_step >= 0:
                        prev_price = self.data.iloc[prev_step]['close']
                        if not pd.isna(prev_price) and prev_price > 0:
                            # Only log once per episode to reduce spam
                            if not hasattr(self, '_logged_fallback_this_episode'):
                                self.logger.warning(f"Invalid price at step {self.current_step}, using price from step {prev_step}: {prev_price:.4f}")
                                self._logged_fallback_this_episode = True
                            return float(prev_price)
                
                # If no valid recent prices, use rolling window average
                window_start = max(0, self.current_step - 20)
                window_data = self.data.iloc[window_start:self.current_step]['close']
                valid_prices = window_data[window_data > 0]
                
                if len(valid_prices) > 0:
                    rolling_avg = valid_prices.mean()
                    if not hasattr(self, '_logged_rolling_this_episode'):
                        self.logger.warning(f"Using rolling average price: {rolling_avg:.4f}")
                        self._logged_rolling_this_episode = True
                    return float(rolling_avg)
                
                # Last resort: global mean (this should rarely happen now)
                global_mean = self.data['close'][self.data['close'] > 0].mean()
                if not hasattr(self, '_logged_global_this_episode'):
                    self.logger.error(f"Data quality issue: using global mean price {global_mean:.4f}")
                    self._logged_global_this_episode = True
                return float(global_mean)
                
            return float(price)
            
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error getting current price: {e}")
            return self.data['close'][self.data['close'] > 0].mean()

    def render(self, mode: str = 'human'):
        """Render the environment (optional)"""
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]['close']
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares_held}")
            print(f"Net Worth: ${self.net_worth:.2f}")
            print(f"Portfolio Return: {((self.net_worth - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            print("-" * 40)
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.portfolio_history:
            return {}
            
        portfolio_values = [p['net_worth'] for p in self.portfolio_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        
        metrics = {
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'volatility': np.std(returns) if len(returns) > 1 else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: list) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0