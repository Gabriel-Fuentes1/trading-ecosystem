"""
Database utilities for storing and retrieving trading data.
Optimized for TimescaleDB with proper connection pooling and error handling.
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import json

from utils.vault_utils import VaultManager

class DatabaseManager:
    """Manage database connections and operations for trading data."""
    
    def __init__(self, pool_size: int = 5, max_connections: int = 20):
        self.vault_manager = VaultManager()
        self.connection_pool = None
        self.pool_size = pool_size
        self.max_connections = max_connections
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self) -> None:
        """Initialize connection pool with credentials from Vault."""
        try:
            # Get database credentials from Vault
            credentials = self.vault_manager.get_secret('trading/data/database')
            
            connection_params = {
                'host': credentials.get('host', 'timescaledb'),
                'port': credentials.get('port', 5432),
                'database': credentials.get('database', 'trading_db'),
                'user': credentials.get('username', 'trading_user'),
                'password': credentials.get('password'),
                'sslmode': 'prefer',
                'connect_timeout': 10,
                'application_name': 'trading_etl'
            }
            
            # Create connection pool
            self.connection_pool = ThreadedConnectionPool(
                minconn=self.pool_size,
                maxconn=self.max_connections,
                **connection_params
            )
            
            logging.info("✅ Database connection pool initialized")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool."""
        if not self.connection_pool:
            raise Exception("Connection pool not initialized")
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool."""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    def store_market_data(self, market_data: List[Dict[str, Any]]) -> int:
        """
        Store market data in TimescaleDB.
        
        :param market_data: List of market data records
        :return: Number of records stored
        """
        if not market_data:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Prepare data for insertion
            insert_query = """
                INSERT INTO market_data (
                    time, symbol, exchange, open, high, low, close, volume, vwap, trades
                ) VALUES %s
                ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    trades = EXCLUDED.trades
            """
            
            # Convert data to tuples
            data_tuples = []
            for record in market_data:
                data_tuple = (
                    record['timestamp'],
                    record['symbol'],
                    record['exchange'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume'],
                    record.get('vwap'),
                    record.get('trades')
                )
                data_tuples.append(data_tuple)
            
            # Execute batch insert
            execute_values(cursor, insert_query, data_tuples, page_size=1000)
            conn.commit()
            
            records_count = len(data_tuples)
            logging.info(f"✅ Stored {records_count} market data records")
            
            return records_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error storing market data: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def store_order_book(self, orderbook_data: List[Dict[str, Any]]) -> int:
        """Store order book data in TimescaleDB."""
        if not orderbook_data:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO order_book (
                    time, symbol, exchange, side, price, quantity, level
                ) VALUES %s
                ON CONFLICT (time, symbol, exchange, side, level) DO UPDATE SET
                    price = EXCLUDED.price,
                    quantity = EXCLUDED.quantity
            """
            
            data_tuples = []
            for record in orderbook_data:
                data_tuple = (
                    record['timestamp'],
                    record['symbol'],
                    record['exchange'],
                    record['side'],
                    record['price'],
                    record['quantity'],
                    record['level']
                )
                data_tuples.append(data_tuple)
            
            execute_values(cursor, insert_query, data_tuples, page_size=1000)
            conn.commit()
            
            records_count = len(data_tuples)
            logging.info(f"✅ Stored {records_count} order book records")
            
            return records_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error storing order book data: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def store_trades(self, trades_data: List[Dict[str, Any]]) -> int:
        """Store trades data in TimescaleDB."""
        if not trades_data:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO trades (
                    time, symbol, exchange, side, price, quantity, trade_id
                ) VALUES %s
                ON CONFLICT (time, symbol, exchange, trade_id) DO NOTHING
            """
            
            data_tuples = []
            for record in trades_data:
                data_tuple = (
                    record['timestamp'],
                    record['symbol'],
                    record['exchange'],
                    record['side'],
                    record['price'],
                    record['quantity'],
                    record.get('trade_id')
                )
                data_tuples.append(data_tuple)
            
            execute_values(cursor, insert_query, data_tuples, page_size=1000)
            conn.commit()
            
            records_count = len(data_tuples)
            logging.info(f"✅ Stored {records_count} trade records")
            
            return records_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error storing trades data: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def store_news_sentiment(self, news_data: List[Dict[str, Any]], sentiment_data: Dict[str, Any]) -> int:
        """Store news data with sentiment analysis results."""
        if not news_data or not sentiment_data.get('sentiment_data'):
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO news_sentiment (
                    time, symbol, title, content, source, sentiment_score, 
                    sentiment_label, confidence, keywords
                ) VALUES %s
                ON CONFLICT (time, title, source) DO UPDATE SET
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_label = EXCLUDED.sentiment_label,
                    confidence = EXCLUDED.confidence
            """
            
            # Create mapping of articles to sentiment
            sentiment_map = {item.get('article_id'): item for item in sentiment_data['sentiment_data']}
            
            data_tuples = []
            for article in news_data:
                article_id = article.get('id')
                sentiment = sentiment_map.get(article_id, {})
                
                data_tuple = (
                    article.get('timestamp', datetime.utcnow()),
                    article.get('symbol'),
                    article.get('title', ''),
                    article.get('content', ''),
                    article.get('source', ''),
                    sentiment.get('sentiment_score'),
                    sentiment.get('sentiment_label'),
                    sentiment.get('confidence'),
                    article.get('keywords', [])
                )
                data_tuples.append(data_tuple)
            
            execute_values(cursor, insert_query, data_tuples, page_size=1000)
            conn.commit()
            
            records_count = len(data_tuples)
            logging.info(f"✅ Stored {records_count} news sentiment records")
            
            return records_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error storing news sentiment data: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def store_onchain_data(self, onchain_data: List[Dict[str, Any]]) -> int:
        """Store on-chain data in TimescaleDB."""
        if not onchain_data:
            return 0
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO onchain_data (
                    time, symbol, metric_name, metric_value, source
                ) VALUES %s
                ON CONFLICT (time, symbol, metric_name, source) DO UPDATE SET
                    metric_value = EXCLUDED.metric_value
            """
            
            data_tuples = []
            for record in onchain_data:
                data_tuple = (
                    record.get('timestamp', datetime.utcnow()),
                    record['symbol'],
                    record['metric_name'],
                    record['metric_value'],
                    record['source']
                )
                data_tuples.append(data_tuple)
            
            execute_values(cursor, insert_query, data_tuples, page_size=1000)
            conn.commit()
            
            records_count = len(data_tuples)
            logging.info(f"✅ Stored {records_count} on-chain data records")
            
            return records_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error storing on-chain data: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def cleanup_old_data(self, table_name: str, days: int = None, hours: int = None) -> int:
        """Clean up old data based on retention policy."""
        if not days and not hours:
            raise ValueError("Either days or hours must be specified")
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Calculate cutoff time
            if days:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
            else:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Delete old records
            delete_query = f"DELETE FROM {table_name} WHERE time < %s"
            cursor.execute(delete_query, (cutoff_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logging.info(f"🧹 Cleaned up {deleted_count} old records from {table_name}")
            
            return deleted_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Error cleaning up {table_name}: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_latest_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest market data for a symbol."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM market_data 
                WHERE symbol = %s 
                ORDER BY time DESC 
                LIMIT %s
            """
            
            cursor.execute(query, (symbol, limit))
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logging.error(f"❌ Error getting latest market data for {symbol}: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def __del__(self):
        """Clean up connection pool on deletion."""
        if self.connection_pool:
            self.connection_pool.closeall()
