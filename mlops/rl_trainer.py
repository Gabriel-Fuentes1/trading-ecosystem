"""
Reinforcement Learning Trainer for Trading Strategies
Implements advanced RL training with backtesting, optimization, and drift detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML and RL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Backtesting and optimization
import vectorbt as vbt
import optuna
from optuna.integration import MLflowCallback

# Explainability and drift detection
import shap
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import DataDriftMetric, DataQualityMetric

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Custom utilities
import sys
import os
sys.path.append('/app')
from utils.database_utils import DatabaseManager
from utils.vault_utils import VaultManager
from environments.trading_env import TradingEnvironment
from backtesting.vectorbt_backtester import VectorBTBacktester
from monitoring.drift_detector import DriftDetector

class RLTrainer:
    """Advanced Reinforcement Learning trainer for trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = DatabaseManager()
        self.vault_manager = VaultManager()
        self.drift_detector = DriftDetector()
        
        # MLflow setup
        self.mlflow_client = MlflowClient()
        self.experiment_name = config.get('experiment_name', 'trading_rl_experiment')
        self._setup_mlflow()
        
        # Training parameters
        self.lookback_window = config.get('lookback_window', 100)
        self.symbols = config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.train_start = config.get('train_start', '2023-01-01')
        self.train_end = config.get('train_end', '2024-01-01')
        self.test_start = config.get('test_start', '2024-01-01')
        self.test_end = config.get('test_end', '2024-12-01')
        
        # Model parameters
        self.algorithm = config.get('algorithm', 'PPO')
        self.total_timesteps = config.get('total_timesteps', 100000)
        self.n_trials = config.get('n_trials', 50)
        
        logging.info(f"Initialized RLTrainer with algorithm: {self.algorithm}")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("http://mlflow:5000")
            
            # Create or get experiment
            try:
                experiment = self.mlflow_client.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(self.experiment_name)
            logging.info(f"MLflow experiment setup: {self.experiment_name}")
            
        except Exception as e:
            logging.error(f"Error setting up MLflow: {e}")
            raise
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and testing data from database."""
        logging.info("Preparing training and testing data...")
        
        try:
            # Get market data for all symbols
            train_data_list = []
            test_data_list = []
            
            for symbol in self.symbols:
                # Get training data
                train_df = self._get_market_data(symbol, self.train_start, self.train_end)
                if not train_df.empty:
                    train_df['symbol'] = symbol
                    train_data_list.append(train_df)
                
                # Get testing data
                test_df = self._get_market_data(symbol, self.test_start, self.test_end)
                if not test_df.empty:
                    test_df['symbol'] = symbol
                    test_data_list.append(test_df)
            
            # Combine data
            train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
            test_data = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()
            
            # Feature engineering
            train_data = self._engineer_features(train_data)
            test_data = self._engineer_features(test_data)
            
            logging.info(f"Training data shape: {train_data.shape}")
            logging.info(f"Testing data shape: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise
    
    def _get_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get market data from database for a specific symbol and date range."""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            
            query = """
                SELECT time, symbol, open, high, low, close, volume
                FROM market_data 
                WHERE symbol = %s 
                AND time >= %s 
                AND time <= %s
                ORDER BY time ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                self.db_manager.return_connection(conn)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for RL training."""
        if df.empty:
            return df
        
        try:
            # Technical indicators using vectorbt
            for symbol in df['symbol'].unique():
                symbol_mask = df['symbol'] == symbol
                symbol_data = df[symbol_mask].copy()
                
                if len(symbol_data) < self.lookback_window:
                    continue
                
                # Price-based features
                symbol_data['returns'] = symbol_data['close'].pct_change()
                symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
                
                # Moving averages
                symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
                symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
                symbol_data['ema_12'] = symbol_data['close'].ewm(span=12).mean()
                symbol_data['ema_26'] = symbol_data['close'].ewm(span=26).mean()
                
                # Volatility
                symbol_data['volatility'] = symbol_data['returns'].rolling(20).std()
                symbol_data['atr'] = self._calculate_atr(symbol_data)
                
                # Momentum indicators
                symbol_data['rsi'] = self._calculate_rsi(symbol_data['close'])
                symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
                
                # Volume indicators
                symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
                symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma']
                
                # Price position indicators
                symbol_data['price_position'] = (symbol_data['close'] - symbol_data['low'].rolling(14).min()) / \
                                              (symbol_data['high'].rolling(14).max() - symbol_data['low'].rolling(14).min())
                
                # Update main dataframe
                df.loc[symbol_mask] = symbol_data
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            logging.info(f"Feature engineering completed. Features: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def optimize_hyperparameters(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logging.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 10),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True),
            }
            
            try:
                # Create environment
                env = TradingEnvironment(
                    data=train_data,
                    lookback_window=self.lookback_window,
                    initial_balance=10000,
                    transaction_cost=0.001
                )
                
                # Create model
                if self.algorithm == 'PPO':
                    model = PPO(
                        'MlpPolicy',
                        env,
                        learning_rate=params['learning_rate'],
                        n_steps=params['n_steps'],
                        batch_size=params['batch_size'],
                        n_epochs=params['n_epochs'],
                        gamma=params['gamma'],
                        gae_lambda=params['gae_lambda'],
                        clip_range=params['clip_range'],
                        ent_coef=params['ent_coef'],
                        verbose=0
                    )
                else:
                    raise ValueError(f"Algorithm {self.algorithm} not supported in optimization")
                
                # Train model
                model.learn(total_timesteps=self.total_timesteps // 4)  # Shorter training for optimization
                
                # Evaluate model
                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
                
                return mean_reward
                
            except Exception as e:
                logging.error(f"Error in optimization trial: {e}")
                return -np.inf
        
        # Create study with MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri="http://mlflow:5000",
            metric_name="mean_reward"
        )
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'rl_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        study.optimize(objective, n_trials=self.n_trials, callbacks=[mlflow_callback])
        
        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")
        logging.info(f"Best score: {study.best_value}")
        
        return best_params
    
    def train_model(self, train_data: pd.DataFrame, best_params: Dict[str, Any]) -> Any:
        """Train the RL model with optimized hyperparameters."""
        logging.info("Training RL model...")
        
        with mlflow.start_run(run_name=f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                # Log parameters
                mlflow.log_params(best_params)
                mlflow.log_params({
                    'algorithm': self.algorithm,
                    'total_timesteps': self.total_timesteps,
                    'lookback_window': self.lookback_window,
                    'symbols': ','.join(self.symbols)
                })
                
                # Create environment
                env = TradingEnvironment(
                    data=train_data,
                    lookback_window=self.lookback_window,
                    initial_balance=10000,
                    transaction_cost=0.001
                )
                
                # Create model with best parameters
                if self.algorithm == 'PPO':
                    model = PPO(
                        'MlpPolicy',
                        env,
                        **best_params,
                        verbose=1
                    )
                elif self.algorithm == 'A2C':
                    model = A2C(
                        'MlpPolicy',
                        env,
                        learning_rate=best_params.get('learning_rate', 3e-4),
                        gamma=best_params.get('gamma', 0.99),
                        verbose=1
                    )
                else:
                    raise ValueError(f"Algorithm {self.algorithm} not supported")
                
                # Custom callback for logging
                callback = MLflowLoggingCallback()
                
                # Train model
                model.learn(
                    total_timesteps=self.total_timesteps,
                    callback=callback
                )
                
                # Save model
                model_path = f"/tmp/rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model.save(model_path)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=f"trading_rl_{self.algorithm.lower()}"
                )
                
                logging.info("Model training completed and logged to MLflow")
                return model
                
            except Exception as e:
                logging.error(f"Error training model: {e}")
                mlflow.log_param("error", str(e))
                raise
    
    def backtest_strategy(self, model: Any, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the trained strategy using VectorBT."""
        logging.info("Starting backtesting with VectorBT...")
        
        try:
            # Initialize backtester
            backtester = VectorBTBacktester(
                initial_cash=10000,
                commission=0.001,
                slippage=0.0005
            )
            
            # Generate signals using the trained model
            signals = self._generate_signals(model, test_data)
            
            # Run backtest
            backtest_results = backtester.run_backtest(
                data=test_data,
                signals=signals,
                symbols=self.symbols
            )
            
            # Calculate advanced metrics
            metrics = self._calculate_advanced_metrics(backtest_results)
            
            # Log metrics to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metrics(metrics)
                
                # Create and log performance plots
                self._create_performance_plots(backtest_results)
            
            logging.info(f"Backtesting completed. Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            return backtest_results
            
        except Exception as e:
            logging.error(f"Error in backtesting: {e}")
            raise
    
    def _generate_signals(self, model: Any, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using the trained model."""
        signals = []
        
        # Create environment for inference
        env = TradingEnvironment(
            data=data,
            lookback_window=self.lookback_window,
            initial_balance=10000,
            transaction_cost=0.001
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record signal
            signals.append({
                'timestamp': env.current_time,
                'symbol': env.current_symbol,
                'action': action,
                'price': env.current_price,
                'position': info.get('position', 0)
            })
        
        return pd.DataFrame(signals)
    
    def _calculate_advanced_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        portfolio_value = backtest_results.get('portfolio_value', pd.Series())
        returns = backtest_results.get('returns', pd.Series())
        
        if portfolio_value.empty or returns.empty:
            return {}
        
        # Basic metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
    
    def _create_performance_plots(self, backtest_results: Dict[str, Any]) -> None:
        """Create and log performance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            
            portfolio_value = backtest_results.get('portfolio_value', pd.Series())
            
            if portfolio_value.empty:
                return
            
            # Portfolio value plot
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_value.index, portfolio_value.values)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.tight_layout()
            
            plot_path = "/tmp/portfolio_performance.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            # Drawdown plot
            rolling_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - rolling_max) / rolling_max
            
            plt.figure(figsize=(12, 4))
            plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            plt.title('Drawdown Over Time')
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.tight_layout()
            
            drawdown_path = "/tmp/drawdown.png"
            plt.savefig(drawdown_path)
            mlflow.log_artifact(drawdown_path)
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating performance plots: {e}")
    
    def explain_model(self, model: Any, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate model explanations using SHAP."""
        logging.info("Generating model explanations with SHAP...")
        
        try:
            # Prepare data for SHAP
            feature_data = self._prepare_shap_data(test_data)
            
            if feature_data.empty:
                return {}
            
            # Create SHAP explainer
            # Note: For RL models, we need to create a wrapper function
            def model_predict(X):
                predictions = []
                for row in X:
                    action, _ = model.predict(row.reshape(1, -1), deterministic=True)
                    predictions.append(action)
                return np.array(predictions)
            
            # Use a subset of data for SHAP (it can be computationally expensive)
            sample_size = min(100, len(feature_data))
            sample_data = feature_data.sample(n=sample_size, random_state=42)
            
            explainer = shap.Explainer(model_predict, sample_data)
            shap_values = explainer(sample_data)
            
            # Create SHAP plots
            with mlflow.start_run(nested=True):
                # Summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, sample_data, show=False)
                shap_summary_path = "/tmp/shap_summary.png"
                plt.savefig(shap_summary_path, bbox_inches='tight')
                mlflow.log_artifact(shap_summary_path)
                plt.close()
                
                # Feature importance
                feature_importance = np.abs(shap_values.values).mean(0)
                importance_dict = dict(zip(sample_data.columns, feature_importance))
                
                # Log feature importance
                mlflow.log_dict(importance_dict, "feature_importance.json")
            
            logging.info("SHAP analysis completed")
            return {
                'feature_importance': importance_dict,
                'shap_values': shap_values
            }
            
        except Exception as e:
            logging.error(f"Error in SHAP analysis: {e}")
            return {}
    
    def _prepare_shap_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for SHAP analysis."""
        # Select numerical features only
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in ['time', 'symbol']]
        
        return data[feature_cols].fillna(0)
    
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data."""
        logging.info("Detecting data drift...")
        
        try:
            drift_results = self.drift_detector.detect_drift(
                reference_data=reference_data,
                current_data=current_data,
                features=self._get_feature_columns(reference_data)
            )
            
            # Log drift results to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    'drift_detected': int(drift_results.get('drift_detected', False)),
                    'drift_score': drift_results.get('drift_score', 0.0),
                    'num_drifted_features': drift_results.get('num_drifted_features', 0)
                })
                
                # Log detailed drift report
                if drift_results.get('drift_report'):
                    mlflow.log_dict(drift_results['drift_report'], "drift_report.json")
            
            logging.info(f"Drift detection completed. Drift detected: {drift_results.get('drift_detected', False)}")
            return drift_results
            
        except Exception as e:
            logging.error(f"Error in drift detection: {e}")
            return {}
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get feature column names."""
        exclude_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in data.columns if col not in exclude_cols]
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logging.info("Starting full training pipeline...")
        
        try:
            # 1. Prepare data
            train_data, test_data = self.prepare_data()
            
            if train_data.empty or test_data.empty:
                raise ValueError("No data available for training")
            
            # 2. Optimize hyperparameters
            best_params = self.optimize_hyperparameters(train_data)
            
            # 3. Train model
            model = self.train_model(train_data, best_params)
            
            # 4. Backtest strategy
            backtest_results = self.backtest_strategy(model, test_data)
            
            # 5. Explain model
            explanation_results = self.explain_model(model, test_data)
            
            # 6. Detect drift
            drift_results = self.detect_drift(train_data, test_data)
            
            # 7. Store results in database
            self._store_training_results({
                'model_path': f"models/rl_{self.algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'hyperparameters': best_params,
                'backtest_metrics': self._calculate_advanced_metrics(backtest_results),
                'drift_detected': drift_results.get('drift_detected', False),
                'training_completed_at': datetime.utcnow()
            })
            
            logging.info("Full training pipeline completed successfully")
            
            return {
                'status': 'success',
                'model': model,
                'backtest_results': backtest_results,
                'explanation_results': explanation_results,
                'drift_results': drift_results
            }
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _store_training_results(self, results: Dict[str, Any]) -> None:
        """Store training results in database."""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO model_training_runs (
                    run_id, experiment_name, model_name, model_version,
                    parameters, metrics, artifacts_path, status,
                    started_at, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else str(datetime.utcnow())
            
            cursor.execute(insert_query, (
                run_id,
                self.experiment_name,
                f"trading_rl_{self.algorithm.lower()}",
                "1.0",
                json.dumps(results.get('hyperparameters', {})),
                json.dumps(results.get('backtest_metrics', {})),
                results.get('model_path', ''),
                'completed',
                results.get('training_started_at', datetime.utcnow()),
                results.get('training_completed_at', datetime.utcnow())
            ))
            
            conn.commit()
            logging.info("Training results stored in database")
            
        except Exception as e:
            logging.error(f"Error storing training results: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.db_manager.return_connection(conn)

class MLflowLoggingCallback(BaseCallback):
    """Custom callback for logging training metrics to MLflow."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        episode_reward = info['episode']['r']
                        episode_length = info['episode']['l']
                        
                        mlflow.log_metric('episode_reward', episode_reward, step=self.n_calls)
                        mlflow.log_metric('episode_length', episode_length, step=self.n_calls)
                        
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        
                        # Log running averages
                        if len(self.episode_rewards) >= 10:
                            avg_reward = np.mean(self.episode_rewards[-10:])
                            mlflow.log_metric('avg_episode_reward_10', avg_reward, step=self.n_calls)
        
        return True

if __name__ == "__main__":
    # Example usage
    config = {
        'experiment_name': 'trading_rl_experiment',
        'algorithm': 'PPO',
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'lookback_window': 100,
        'total_timesteps': 100000,
        'n_trials': 20,
        'train_start': '2023-01-01',
        'train_end': '2024-01-01',
        'test_start': '2024-01-01',
        'test_end': '2024-12-01'
    }
    
    trainer = RLTrainer(config)
    results = trainer.run_full_training_pipeline()
    
    if results['status'] == 'success':
        print("Training completed successfully!")
    else:
        print(f"Training failed: {results.get('error', 'Unknown error')}")
