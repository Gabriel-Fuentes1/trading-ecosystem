"""
Market Data Ingestion DAG
Orchestrates the collection of market data from multiple sources including
Binance, Polygon.io, and alternative data sources.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

# Import custom operators and utilities
import sys
import os
sys.path.append('/opt/airflow/plugins')

from operators.data_validation_operator import DataValidationOperator
from operators.sentiment_analysis_operator import SentimentAnalysisOperator
from utils.database_utils import DatabaseManager
from utils.cache_utils import CacheManager
from utils.vault_utils import VaultManager

# DAG Configuration
DAG_ID = 'market_data_ingestion'
SCHEDULE_INTERVAL = '*/5 * * * *'  # Every 5 minutes
MAX_ACTIVE_RUNS = 1
CATCHUP = False

# Default arguments
default_args = {
    'owner': 'trading-system',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=10),
}

# Initialize DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Ingest market data from multiple sources with validation and caching',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=MAX_ACTIVE_RUNS,
    catchup=CATCHUP,
    tags=['market-data', 'ingestion', 'real-time'],
    doc_md=__doc__,
)

def get_trading_symbols(**context) -> List[str]:
    """Get list of symbols to trade from configuration."""
    try:
        # Get symbols from Airflow Variables or use defaults
        symbols_str = Variable.get('trading_symbols', default_var='BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT')
        symbols = [s.strip() for s in symbols_str.split(',')]
        
        logging.info(f"Trading symbols: {symbols}")
        return symbols
    except Exception as e:
        logging.error(f"Error getting trading symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT']  # Fallback symbols

def ingest_binance_data(**context) -> Dict[str, Any]:
    """Ingest real-time market data from Binance."""
    from plugins.extractors.binance_extractor import BinanceExtractor
    
    symbols = context['task_instance'].xcom_pull(task_ids='get_trading_symbols')
    extractor = BinanceExtractor()
    
    try:
        # Extract OHLCV data
        market_data = extractor.get_klines(symbols, interval='5m', limit=100)
        
        # Extract order book data
        orderbook_data = extractor.get_order_book(symbols, limit=100)
        
        # Extract recent trades
        trades_data = extractor.get_recent_trades(symbols, limit=500)
        
        result = {
            'market_data': market_data,
            'orderbook_data': orderbook_data,
            'trades_data': trades_data,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'binance',
            'symbols_count': len(symbols)
        }
        
        logging.info(f"Successfully ingested Binance data for {len(symbols)} symbols")
        return result
        
    except Exception as e:
        logging.error(f"Error ingesting Binance data: {e}")
        raise

def ingest_polygon_data(**context) -> Dict[str, Any]:
    """Ingest market data from Polygon.io for traditional assets."""
    from plugins.extractors.polygon_extractor import PolygonExtractor
    
    # Get traditional symbols (stocks, forex)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']  # Example symbols
    extractor = PolygonExtractor()
    
    try:
        # Extract aggregates (bars) data
        market_data = extractor.get_aggregates(symbols, timespan='minute', multiplier=5)
        
        # Extract trades data
        trades_data = extractor.get_trades(symbols)
        
        result = {
            'market_data': market_data,
            'trades_data': trades_data,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'polygon',
            'symbols_count': len(symbols)
        }
        
        logging.info(f"Successfully ingested Polygon data for {len(symbols)} symbols")
        return result
        
    except Exception as e:
        logging.error(f"Error ingesting Polygon data: {e}")
        raise

def ingest_news_data(**context) -> Dict[str, Any]:
    """Ingest news data for sentiment analysis."""
    from plugins.extractors.news_extractor import NewsExtractor
    
    symbols = context['task_instance'].xcom_pull(task_ids='get_trading_symbols')
    extractor = NewsExtractor()
    
    try:
        # Extract news articles
        news_data = extractor.get_crypto_news(symbols, hours_back=1)
        
        result = {
            'news_data': news_data,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'news_api',
            'articles_count': len(news_data)
        }
        
        logging.info(f"Successfully ingested {len(news_data)} news articles")
        return result
        
    except Exception as e:
        logging.error(f"Error ingesting news data: {e}")
        raise

def ingest_onchain_data(**context) -> Dict[str, Any]:
    """Ingest on-chain data for crypto assets."""
    from plugins.extractors.onchain_extractor import OnChainExtractor
    
    symbols = context['task_instance'].xcom_pull(task_ids='get_trading_symbols')
    extractor = OnChainExtractor()
    
    try:
        # Extract on-chain metrics
        onchain_data = extractor.get_metrics(symbols)
        
        result = {
            'onchain_data': onchain_data,
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'onchain',
            'metrics_count': len(onchain_data)
        }
        
        logging.info(f"Successfully ingested on-chain data for {len(symbols)} symbols")
        return result
        
    except Exception as e:
        logging.error(f"Error ingesting on-chain data: {e}")
        raise

def store_market_data(**context) -> Dict[str, Any]:
    """Store validated market data in TimescaleDB."""
    db_manager = DatabaseManager()
    cache_manager = CacheManager()
    
    # Get data from previous tasks
    binance_data = context['task_instance'].xcom_pull(task_ids='binance_ingestion.ingest_binance_data')
    polygon_data = context['task_instance'].xcom_pull(task_ids='polygon_ingestion.ingest_polygon_data')
    
    try:
        stored_records = 0
        
        # Store Binance market data
        if binance_data and binance_data.get('market_data'):
            records = db_manager.store_market_data(binance_data['market_data'])
            stored_records += records
            
            # Cache recent data for fast access
            cache_manager.cache_market_data(binance_data['market_data'])
        
        # Store Polygon market data
        if polygon_data and polygon_data.get('market_data'):
            records = db_manager.store_market_data(polygon_data['market_data'])
            stored_records += records
            
            # Cache recent data
            cache_manager.cache_market_data(polygon_data['market_data'])
        
        result = {
            'stored_records': stored_records,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success'
        }
        
        logging.info(f"Successfully stored {stored_records} market data records")
        return result
        
    except Exception as e:
        logging.error(f"Error storing market data: {e}")
        raise

def store_alternative_data(**context) -> Dict[str, Any]:
    """Store alternative data (news, on-chain) in database."""
    db_manager = DatabaseManager()
    
    # Get data from previous tasks
    news_data = context['task_instance'].xcom_pull(task_ids='alternative_data.ingest_news_data')
    onchain_data = context['task_instance'].xcom_pull(task_ids='alternative_data.ingest_onchain_data')
    sentiment_data = context['task_instance'].xcom_pull(task_ids='alternative_data.analyze_sentiment')
    
    try:
        stored_records = 0
        
        # Store news data with sentiment
        if news_data and sentiment_data:
            records = db_manager.store_news_sentiment(news_data['news_data'], sentiment_data)
            stored_records += records
        
        # Store on-chain data
        if onchain_data and onchain_data.get('onchain_data'):
            records = db_manager.store_onchain_data(onchain_data['onchain_data'])
            stored_records += records
        
        result = {
            'stored_records': stored_records,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success'
        }
        
        logging.info(f"Successfully stored {stored_records} alternative data records")
        return result
        
    except Exception as e:
        logging.error(f"Error storing alternative data: {e}")
        raise

def cleanup_old_data(**context) -> Dict[str, Any]:
    """Clean up old data based on retention policies."""
    db_manager = DatabaseManager()
    
    try:
        # Clean up old order book data (keep only 24 hours)
        deleted_orderbook = db_manager.cleanup_old_data('order_book', hours=24)
        
        # Clean up old system logs (keep only 7 days)
        deleted_logs = db_manager.cleanup_old_data('system_logs', days=7)
        
        result = {
            'deleted_orderbook_records': deleted_orderbook,
            'deleted_log_records': deleted_logs,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success'
        }
        
        logging.info(f"Cleanup completed: {deleted_orderbook + deleted_logs} records deleted")
        return result
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        raise

# Task definitions
get_symbols_task = PythonOperator(
    task_id='get_trading_symbols',
    python_callable=get_trading_symbols,
    dag=dag,
    doc_md="Get the list of trading symbols from configuration"
)

# Binance data ingestion task group
with TaskGroup('binance_ingestion', dag=dag) as binance_group:
    ingest_binance_task = PythonOperator(
        task_id='ingest_binance_data',
        python_callable=ingest_binance_data,
        dag=dag,
        pool='data_ingestion_pool',
        doc_md="Ingest real-time market data from Binance API"
    )
    
    validate_binance_task = DataValidationOperator(
        task_id='validate_binance_data',
        data_source='binance',
        validation_schema='market_data_schema',
        dag=dag,
        doc_md="Validate Binance market data using Pandera schemas"
    )
    
    ingest_binance_task >> validate_binance_task

# Polygon data ingestion task group
with TaskGroup('polygon_ingestion', dag=dag) as polygon_group:
    ingest_polygon_task = PythonOperator(
        task_id='ingest_polygon_data',
        python_callable=ingest_polygon_data,
        dag=dag,
        pool='data_ingestion_pool',
        doc_md="Ingest market data from Polygon.io API"
    )
    
    validate_polygon_task = DataValidationOperator(
        task_id='validate_polygon_data',
        data_source='polygon',
        validation_schema='market_data_schema',
        dag=dag,
        doc_md="Validate Polygon market data using Pandera schemas"
    )
    
    ingest_polygon_task >> validate_polygon_task

# Alternative data ingestion task group
with TaskGroup('alternative_data', dag=dag) as alternative_group:
    ingest_news_task = PythonOperator(
        task_id='ingest_news_data',
        python_callable=ingest_news_data,
        dag=dag,
        pool='data_ingestion_pool',
        doc_md="Ingest news data for sentiment analysis"
    )
    
    ingest_onchain_task = PythonOperator(
        task_id='ingest_onchain_data',
        python_callable=ingest_onchain_data,
        dag=dag,
        pool='data_ingestion_pool',
        doc_md="Ingest on-chain metrics for crypto assets"
    )
    
    analyze_sentiment_task = SentimentAnalysisOperator(
        task_id='analyze_sentiment',
        model_name='finbert',
        dag=dag,
        doc_md="Analyze sentiment of news articles using FinBERT"
    )
    
    validate_alternative_task = DataValidationOperator(
        task_id='validate_alternative_data',
        data_source='alternative',
        validation_schema='alternative_data_schema',
        dag=dag,
        doc_md="Validate alternative data using Pandera schemas"
    )
    
    [ingest_news_task, ingest_onchain_task] >> analyze_sentiment_task >> validate_alternative_task

# Data storage tasks
store_market_data_task = PythonOperator(
    task_id='store_market_data',
    python_callable=store_market_data,
    dag=dag,
    pool='database_pool',
    doc_md="Store validated market data in TimescaleDB"
)

store_alternative_data_task = PythonOperator(
    task_id='store_alternative_data',
    python_callable=store_alternative_data,
    dag=dag,
    pool='database_pool',
    doc_md="Store alternative data in TimescaleDB"
)

# Cleanup task
cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=dag,
    trigger_rule='all_done',  # Run even if some tasks fail
    doc_md="Clean up old data based on retention policies"
)

# Data quality monitoring
data_quality_check = BashOperator(
    task_id='data_quality_check',
    bash_command='python /opt/airflow/plugins/monitoring/data_quality_monitor.py',
    dag=dag,
    trigger_rule='all_success',
    doc_md="Run data quality checks and generate alerts"
)

# Task dependencies
get_symbols_task >> [binance_group, polygon_group, alternative_group]

[binance_group, polygon_group] >> store_market_data_task
alternative_group >> store_alternative_data_task

[store_market_data_task, store_alternative_data_task] >> data_quality_check >> cleanup_task
