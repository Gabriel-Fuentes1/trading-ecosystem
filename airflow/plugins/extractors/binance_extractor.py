"""
Binance Data Extractor
Extracts market data, order book, and trades from Binance API.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from utils.vault_utils import VaultManager
from utils.rate_limiter import RateLimiter

class BinanceExtractor:
    """Extract data from Binance API with rate limiting and error handling."""
    
    def __init__(self):
        self.vault_manager = VaultManager()
        self.client = None
        self.rate_limiter = RateLimiter(requests_per_minute=1200)  # Binance limit
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Binance client with credentials from Vault."""
        try:
            # Get credentials from Vault
            credentials = self.vault_manager.get_secret('trading/data/binance')
            
            api_key = credentials.get('api_key')
            api_secret = credentials.get('api_secret')
            testnet = credentials.get('testnet', True)
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in Vault")
            
            # Initialize client
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
            
            # Test connection
            self.client.ping()
            logging.info("✅ Binance client initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Binance client: {e}")
            raise
    
    def get_klines(self, symbols: List[str], interval: str = '5m', limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get OHLCV data (klines) for multiple symbols.
        
        :param symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        :param interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
        :param limit: Number of klines to retrieve (max 1000)
        :return: List of market data records
        """
        market_data = []
        
        for symbol in symbols:
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Get klines from Binance
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                # Convert to standardized format
                for kline in klines:
                    record = {
                        'symbol': symbol,
                        'timestamp': pd.to_datetime(kline[0], unit='ms'),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'close_time': pd.to_datetime(kline[6], unit='ms'),
                        'quote_volume': float(kline[7]),
                        'trades': int(kline[8]),
                        'taker_buy_base_volume': float(kline[9]),
                        'taker_buy_quote_volume': float(kline[10]),
                        'exchange': 'binance',
                        'interval': interval
                    }
                    market_data.append(record)
                
                logging.info(f"✅ Retrieved {len(klines)} klines for {symbol}")
                
            except BinanceAPIException as e:
                logging.error(f"❌ Binance API error for {symbol}: {e}")
                continue
            except Exception as e:
                logging.error(f"❌ Error getting klines for {symbol}: {e}")
                continue
        
        logging.info(f"📊 Total market data records: {len(market_data)}")
        return market_data
    
    def get_order_book(self, symbols: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order book data for multiple symbols.
        
        :param symbols: List of trading symbols
        :param limit: Number of price levels (5, 10, 20, 50, 100, 500, 1000, 5000)
        :return: List of order book records
        """
        orderbook_data = []
        
        for symbol in symbols:
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Get order book
                orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
                
                timestamp = pd.to_datetime(datetime.utcnow())
                
                # Process bids
                for level, (price, quantity) in enumerate(orderbook['bids']):
                    record = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'side': 'bid',
                        'price': float(price),
                        'quantity': float(quantity),
                        'level': level + 1,
                        'exchange': 'binance'
                    }
                    orderbook_data.append(record)
                
                # Process asks
                for level, (price, quantity) in enumerate(orderbook['asks']):
                    record = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'side': 'ask',
                        'price': float(price),
                        'quantity': float(quantity),
                        'level': level + 1,
                        'exchange': 'binance'
                    }
                    orderbook_data.append(record)
                
                logging.info(f"✅ Retrieved order book for {symbol} ({len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks)")
                
            except BinanceAPIException as e:
                logging.error(f"❌ Binance API error for {symbol} order book: {e}")
                continue
            except Exception as e:
                logging.error(f"❌ Error getting order book for {symbol}: {e}")
                continue
        
        logging.info(f"📖 Total order book records: {len(orderbook_data)}")
        return orderbook_data
    
    def get_recent_trades(self, symbols: List[str], limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades for multiple symbols.
        
        :param symbols: List of trading symbols
        :param limit: Number of recent trades (max 1000)
        :return: List of trade records
        """
        trades_data = []
        
        for symbol in symbols:
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Get recent trades
                trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
                
                # Convert to standardized format
                for trade in trades:
                    record = {
                        'symbol': symbol,
                        'timestamp': pd.to_datetime(trade['time'], unit='ms'),
                        'trade_id': str(trade['id']),
                        'price': float(trade['price']),
                        'quantity': float(trade['qty']),
                        'side': 'sell' if trade['isBuyerMaker'] else 'buy',
                        'is_best_match': trade['isBestMatch'],
                        'exchange': 'binance'
                    }
                    trades_data.append(record)
                
                logging.info(f"✅ Retrieved {len(trades)} recent trades for {symbol}")
                
            except BinanceAPIException as e:
                logging.error(f"❌ Binance API error for {symbol} trades: {e}")
                continue
            except Exception as e:
                logging.error(f"❌ Error getting trades for {symbol}: {e}")
                continue
        
        logging.info(f"💱 Total trade records: {len(trades_data)}")
        return trades_data
    
    def get_24hr_ticker(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get 24hr ticker statistics for multiple symbols.
        
        :param symbols: List of trading symbols
        :return: List of ticker records
        """
        ticker_data = []
        
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Get all tickers at once (more efficient)
            tickers = self.client.get_ticker()
            
            # Filter for requested symbols
            symbol_set = set(symbols)
            timestamp = pd.to_datetime(datetime.utcnow())
            
            for ticker in tickers:
                if ticker['symbol'] in symbol_set:
                    record = {
                        'symbol': ticker['symbol'],
                        'timestamp': timestamp,
                        'price_change': float(ticker['priceChange']),
                        'price_change_percent': float(ticker['priceChangePercent']),
                        'weighted_avg_price': float(ticker['weightedAvgPrice']),
                        'prev_close_price': float(ticker['prevClosePrice']),
                        'last_price': float(ticker['lastPrice']),
                        'last_qty': float(ticker['lastQty']),
                        'bid_price': float(ticker['bidPrice']),
                        'ask_price': float(ticker['askPrice']),
                        'open_price': float(ticker['openPrice']),
                        'high_price': float(ticker['highPrice']),
                        'low_price': float(ticker['lowPrice']),
                        'volume': float(ticker['volume']),
                        'quote_volume': float(ticker['quoteVolume']),
                        'open_time': pd.to_datetime(ticker['openTime'], unit='ms'),
                        'close_time': pd.to_datetime(ticker['closeTime'], unit='ms'),
                        'count': int(ticker['count']),
                        'exchange': 'binance'
                    }
                    ticker_data.append(record)
            
            logging.info(f"📈 Retrieved 24hr ticker data for {len(ticker_data)} symbols")
            
        except BinanceAPIException as e:
            logging.error(f"❌ Binance API error getting tickers: {e}")
        except Exception as e:
            logging.error(f"❌ Error getting ticker data: {e}")
        
        return ticker_data
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including trading rules and symbol info."""
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            exchange_info = self.client.get_exchange_info()
            
            logging.info("✅ Retrieved exchange information")
            return exchange_info
            
        except BinanceAPIException as e:
            logging.error(f"❌ Binance API error getting exchange info: {e}")
            return {}
        except Exception as e:
            logging.error(f"❌ Error getting exchange info: {e}")
            return {}
