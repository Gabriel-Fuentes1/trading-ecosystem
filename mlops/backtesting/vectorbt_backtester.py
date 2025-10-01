"""
VectorBT Backtester for Trading Strategies
High-performance backtesting with realistic transaction costs and slippage.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

class VectorBTBacktester:
    """
    Advanced backtesting engine using VectorBT for high-performance analysis.
    """
    
    def __init__(
        self,
        initial_cash: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 1.0
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        logging.info(f"VectorBT Backtester initialized with ${initial_cash} initial cash")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest using VectorBT.
        
        :param data: Market data with OHLCV
        :param signals: Trading signals with timestamps and actions
        :param symbols: List of symbols to backtest
        :return: Backtest results with metrics and portfolio data
        """
        logging.info("Starting VectorBT backtesting...")
        
        try:
            results = {}
            
            for symbol in symbols:
                symbol_data = data[data['symbol'] == symbol].copy()
                symbol_signals = signals[signals['symbol'] == symbol].copy()
                
                if symbol_data.empty or symbol_signals.empty:
                    logging.warning(f"No data or signals for {symbol}, skipping...")
                    continue
                
                # Run backtest for this symbol
                symbol_results = self._backtest_symbol(symbol_data, symbol_signals, symbol)
                results[symbol] = symbol_results
            
            # Combine results across symbols
            combined_results = self._combine_symbol_results(results)
            
            logging.info("VectorBT backtesting completed")
            return combined_results
            
        except Exception as e:
            logging.error(f"Error in VectorBT backtesting: {e}")
            raise
    
    def _backtest_symbol(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Run backtest for a single symbol."""
        try:
            # Prepare data
            data = data.set_index('time').sort_index()
            
            # Create price series
            close_prices = data['close']
            high_prices = data['high']
            low_prices = data['low']
            
            # Convert signals to entries and exits
            entries, exits = self._convert_signals_to_entries_exits(signals, data.index)
            
            # Apply slippage to prices
            entry_prices = self._apply_slippage(close_prices, entries, 'buy')
            exit_prices = self._apply_slippage(close_prices, exits, 'sell')
            
            # Run portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entries,
                exits=exits,
                entry_price=entry_prices,
                exit_price=exit_prices,
                init_cash=self.initial_cash,
                fees=self.commission,
                freq='5T'  # 5-minute frequency
            )
            
            # Calculate metrics
            metrics = self._calculate_symbol_metrics(portfolio, close_prices)
            
            return {
                'portfolio': portfolio,
                'metrics': metrics,
                'close_prices': close_prices,
                'entries': entries,
                'exits': exits,
                'symbol': symbol
            }
            
        except Exception as e:
            logging.error(f"Error backtesting {symbol}: {e}")
            return {}
    
    def _convert_signals_to_entries_exits(
        self,
        signals: pd.DataFrame,
        price_index: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series]:
        """Convert trading signals to entry/exit boolean series."""
        # Initialize boolean series
        entries = pd.Series(False, index=price_index)
        exits = pd.Series(False, index=price_index)
        
        # Convert signals
        for _, signal in signals.iterrows():
            timestamp = signal['timestamp']
            action = signal['action']
            
            # Find closest timestamp in price index
            if timestamp in price_index:
                idx = timestamp
            else:
                # Find nearest timestamp
                idx = price_index[price_index.get_indexer([timestamp], method='nearest')[0]]
            
            if action == 1:  # Buy signal
                entries.loc[idx] = True
            elif action == 2:  # Sell signal
                exits.loc[idx] = True
        
        return entries, exits
    
    def _apply_slippage(
        self,
        prices: pd.Series,
        signals: pd.Series,
        direction: str
    ) -> pd.Series:
        """Apply slippage to execution prices."""
        slippage_factor = 1 + self.slippage if direction == 'buy' else 1 - self.slippage
        
        # Apply slippage only where signals are True
        adjusted_prices = prices.copy()
        adjusted_prices[signals] = prices[signals] * slippage_factor
        
        return adjusted_prices
    
    def _calculate_symbol_metrics(
        self,
        portfolio: vbt.Portfolio,
        close_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for a single symbol."""
        try:
            # Basic metrics
            total_return = portfolio.total_return()
            total_trades = portfolio.trades.count()
            win_rate = portfolio.trades.win_rate() if total_trades > 0 else 0
            
            # Risk metrics
            returns = portfolio.returns()
            sharpe_ratio = returns.sharpe_ratio() if len(returns) > 0 else 0
            sortino_ratio = returns.sortino_ratio() if len(returns) > 0 else 0
            max_drawdown = portfolio.max_drawdown()
            
            # Calmar ratio
            annualized_return = returns.annualized_return() if len(returns) > 0 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade metrics
            avg_trade_duration = portfolio.trades.duration.mean() if total_trades > 0 else 0
            profit_factor = portfolio.trades.profit_factor() if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_trade_duration': avg_trade_duration,
                'profit_factor': profit_factor,
                'final_value': portfolio.final_value()
            }
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            return {}
    
    def _combine_symbol_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple symbols into portfolio-level metrics."""
        if not results:
            return {}
        
        try:
            # Combine portfolio values
            portfolio_values = []
            all_returns = []
            combined_metrics = {}
            
            for symbol, symbol_results in results.items():
                if 'portfolio' in symbol_results:
                    portfolio = symbol_results['portfolio']
                    portfolio_values.append(portfolio.value())
                    all_returns.extend(portfolio.returns().values)
            
            if not portfolio_values:
                return {}
            
            # Calculate combined portfolio value
            combined_portfolio_value = pd.concat(portfolio_values, axis=1).sum(axis=1)
            combined_returns = pd.Series(all_returns)
            
            # Calculate combined metrics
            initial_value = len(results) * self.initial_cash
            final_value = combined_portfolio_value.iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Risk metrics
            if len(combined_returns) > 0:
                sharpe_ratio = combined_returns.mean() / combined_returns.std() * np.sqrt(252 * 24 * 12) if combined_returns.std() > 0 else 0
                
                # Sortino ratio
                negative_returns = combined_returns[combined_returns < 0]
                sortino_ratio = combined_returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 12) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Drawdown
            rolling_max = combined_portfolio_value.expanding().max()
            drawdown = (combined_portfolio_value - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Aggregate trade metrics
            total_trades = sum(result.get('metrics', {}).get('total_trades', 0) for result in results.values())
            avg_win_rate = np.mean([result.get('metrics', {}).get('win_rate', 0) for result in results.values()])
            
            combined_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'num_symbols': len(results)
            }
            
            return {
                'portfolio_value': combined_portfolio_value,
                'returns': combined_returns,
                'metrics': combined_metrics,
                'symbol_results': results,
                'drawdown': drawdown
            }
            
        except Exception as e:
            logging.error(f"Error combining results: {e}")
            return {}
    
    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        symbols: List[str],
        train_period_days: int = 90,
        test_period_days: int = 30,
        step_days: int = 7
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis for more robust backtesting.
        
        :param data: Market data
        :param signals: Trading signals
        :param symbols: List of symbols
        :param train_period_days: Training period length in days
        :param test_period_days: Testing period length in days
        :param step_days: Step size for rolling window in days
        :return: Walk-forward analysis results
        """
        logging.info("Starting walk-forward analysis...")
        
        try:
            # Get date range
            start_date = data['time'].min()
            end_date = data['time'].max()
            
            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(
                start_date, end_date, train_period_days, test_period_days, step_days
            )
            
            walk_forward_results = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logging.info(f"Walk-forward window {i+1}/{len(windows)}: {test_start} to {test_end}")
                
                # Get data for this window
                test_data = data[(data['time'] >= test_start) & (data['time'] <= test_end)]
                test_signals = signals[(signals['timestamp'] >= test_start) & (signals['timestamp'] <= test_end)]
                
                if test_data.empty or test_signals.empty:
                    continue
                
                # Run backtest for this window
                window_results = self.run_backtest(test_data, test_signals, symbols)
                
                if window_results:
                    window_results['window_info'] = {
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'window_number': i + 1
                    }
                    walk_forward_results.append(window_results)
            
            # Aggregate walk-forward results
            aggregated_results = self._aggregate_walk_forward_results(walk_forward_results)
            
            logging.info(f"Walk-forward analysis completed with {len(walk_forward_results)} windows")
            return aggregated_results
            
        except Exception as e:
            logging.error(f"Error in walk-forward analysis: {e}")
            raise
    
    def _generate_walk_forward_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        train_period_days: int,
        test_period_days: int,
        step_days: int
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward analysis windows."""
        windows = []
        current_date = start_date + pd.Timedelta(days=train_period_days)
        
        while current_date + pd.Timedelta(days=test_period_days) <= end_date:
            train_start = current_date - pd.Timedelta(days=train_period_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.Timedelta(days=test_period_days)
            
            windows.append((train_start, train_end, test_start, test_end))
            current_date += pd.Timedelta(days=step_days)
        
        return windows
    
    def _aggregate_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from walk-forward analysis."""
        if not results:
            return {}
        
        try:
            # Collect metrics from all windows
            all_metrics = []
            for result in results:
                if 'metrics' in result:
                    all_metrics.append(result['metrics'])
            
            if not all_metrics:
                return {}
            
            # Calculate aggregate statistics
            metric_names = all_metrics[0].keys()
            aggregated_metrics = {}
            
            for metric in metric_names:
                values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
                if values:
                    aggregated_metrics[f'{metric}_mean'] = np.mean(values)
                    aggregated_metrics[f'{metric}_std'] = np.std(values)
                    aggregated_metrics[f'{metric}_min'] = np.min(values)
                    aggregated_metrics[f'{metric}_max'] = np.max(values)
            
            # Calculate consistency metrics
            returns = [m.get('total_return', 0) for m in all_metrics]
            positive_periods = sum(1 for r in returns if r > 0)
            consistency_ratio = positive_periods / len(returns) if returns else 0
            
            aggregated_metrics['consistency_ratio'] = consistency_ratio
            aggregated_metrics['num_windows'] = len(results)
            
            return {
                'aggregated_metrics': aggregated_metrics,
                'individual_results': results,
                'summary': {
                    'total_windows': len(results),
                    'positive_windows': positive_periods,
                    'consistency_ratio': consistency_ratio
                }
            }
            
        except Exception as e:
            logging.error(f"Error aggregating walk-forward results: {e}")
            return {}
