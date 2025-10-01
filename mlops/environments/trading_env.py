"""
Trading Environment for Reinforcement Learning
Gymnasium-compatible environment for training RL trading agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents.
    
    Action Space: Discrete(3) - 0: Hold, 1: Buy, 2: Sell
    Observation Space: Box - Market features and portfolio state
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback_window: int = 100,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        reward_function: str = 'sharpe'
    ):
        super().__init__()
        
        self.data = data.copy()
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.reward_function = reward_function
        
        # Prepare data
        self._prepare_data()
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position (-1 to 1, where 1 is fully long)
        self.portfolio_value = initial_balance
        self.trade_history = []
        self.returns_history = []
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        # Observation space: market features + portfolio state
        n_features = len(self._get_feature_columns())
        obs_dim = n_features * lookback_window + 4  # +4 for portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logging.info(f"Trading environment initialized with {len(self.data)} data points")
    
    def _prepare_data(self) -> None:
        """Prepare and normalize data for training."""
        # Sort by time
        self.data = self.data.sort_values('time').reset_index(drop=True)
        
        # Get feature columns
        self.feature_columns = self._get_feature_columns()
        
        # Normalize features
        for col in self.feature_columns:
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val > 0:
                    self.data[col] = (self.data[col] - mean_val) / std_val
        
        # Fill any remaining NaN values
        self.data = self.data.fillna(0)
        
        logging.info(f"Data prepared with features: {self.feature_columns}")
    
    def _get_feature_columns(self) -> list:
        """Get list of feature columns for observations."""
        exclude_cols = ['time', 'symbol']
        return [col for col in self.data.columns if col not in exclude_cols]
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.returns_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return reward."""
        old_portfolio_value = self.portfolio_value
        
        # Calculate current portfolio value
        position_value = self.position * current_price * self.initial_balance
        self.portfolio_value = self.balance + position_value
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0 is hold, no action needed
        
        # Calculate reward
        reward = self._calculate_reward(old_portfolio_value, current_price)
        
        # Update returns history
        if old_portfolio_value > 0:
            portfolio_return = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
            self.returns_history.append(portfolio_return)
        
        return reward
    
    def _execute_buy(self, price: float) -> None:
        """Execute buy order."""
        if self.position < self.max_position_size:
            # Calculate how much we can buy
            available_balance = self.balance * 0.95  # Keep some cash
            max_buy_amount = (self.max_position_size - self.position) * self.initial_balance
            buy_amount = min(available_balance, max_buy_amount)
            
            if buy_amount > 0:
                # Calculate position change
                position_change = buy_amount / (price * self.initial_balance)
                
                # Apply transaction cost
                cost = buy_amount * self.transaction_cost
                
                # Update position and balance
                self.position += position_change
                self.balance -= (buy_amount + cost)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': price,
                    'amount': buy_amount,
                    'position': self.position,
                    'balance': self.balance
                })
    
    def _execute_sell(self, price: float) -> None:
        """Execute sell order."""
        if self.position > -self.max_position_size:
            # Calculate how much we can sell
            max_sell_amount = (self.position + self.max_position_size) * self.initial_balance
            sell_amount = min(abs(self.position) * self.initial_balance * price, max_sell_amount)
            
            if sell_amount > 0:
                # Calculate position change
                position_change = sell_amount / (price * self.initial_balance)
                
                # Apply transaction cost
                cost = sell_amount * self.transaction_cost
                
                # Update position and balance
                self.position -= position_change
                self.balance += (sell_amount - cost)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': price,
                    'amount': sell_amount,
                    'position': self.position,
                    'balance': self.balance
                })
    
    def _calculate_reward(self, old_portfolio_value: float, current_price: float) -> float:
        """Calculate reward based on the selected reward function."""
        if self.reward_function == 'return':
            # Simple return-based reward
            if old_portfolio_value > 0:
                return (self.portfolio_value - old_portfolio_value) / old_portfolio_value
            return 0
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio-based reward
            if len(self.returns_history) < 10:
                return 0
            
            recent_returns = np.array(self.returns_history[-10:])
            if np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / np.std(recent_returns)
                return sharpe
            return 0
        
        elif self.reward_function == 'sortino':
            # Sortino ratio-based reward
            if len(self.returns_history) < 10:
                return 0
            
            recent_returns = np.array(self.returns_history[-10:])
            negative_returns = recent_returns[recent_returns < 0]
            
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino = np.mean(recent_returns) / np.std(negative_returns)
                return sortino
            elif np.mean(recent_returns) > 0:
                return np.mean(recent_returns)
            return 0
        
        else:
            # Default to simple return
            if old_portfolio_value > 0:
                return (self.portfolio_value - old_portfolio_value) / old_portfolio_value
            return 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            start_idx = 0
            padding = self.lookback_window - self.current_step
        else:
            start_idx = self.current_step - self.lookback_window
            padding = 0
        
        # Get market features
        end_idx = self.current_step + 1
        market_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values
        
        # Pad if necessary
        if padding > 0:
            pad_shape = (padding, len(self.feature_columns))
            market_data = np.vstack([np.zeros(pad_shape), market_data])
        
        # Flatten market data
        market_features = market_data.flatten()
        
        # Add portfolio state
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 1.0
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            len(self.trade_history) / 100.0  # Normalized trade count
        ])
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_state]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 1.0
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'current_price': current_price,
            'total_trades': len(self.trade_history),
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance
        }
    
    @property
    def current_time(self) -> pd.Timestamp:
        """Get current timestamp."""
        if self.current_step < len(self.data):
            return self.data.iloc[self.current_step]['time']
        return pd.Timestamp.now()
    
    @property
    def current_symbol(self) -> str:
        """Get current symbol."""
        if self.current_step < len(self.data) and 'symbol' in self.data.columns:
            return self.data.iloc[self.current_step]['symbol']
        return 'UNKNOWN'
    
    @property
    def current_price(self) -> float:
        """Get current price."""
        if self.current_step < len(self.data):
            return self.data.iloc[self.current_step]['close']
        return 0.0
    
    def render(self, mode='human') -> None:
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Total Return: {((self.portfolio_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
            print("-" * 50)
