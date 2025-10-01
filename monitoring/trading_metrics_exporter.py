"""
Custom Trading Metrics Exporter
Exports trading-specific metrics to Prometheus.
"""

import asyncio
import logging
import time
from typing import Dict, Any
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import asyncpg
import redis
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
portfolio_value = Gauge('portfolio_value_total', 'Total portfolio value in USD')
portfolio_var_1d = Gauge('portfolio_var_1d', 'Portfolio 1-day Value at Risk')
portfolio_positions = Gauge('portfolio_positions_total', 'Total number of positions')
portfolio_pnl = Gauge('portfolio_pnl_daily', 'Daily P&L')

trading_volume = Gauge('trading_volume_total', 'Total trading volume')
order_execution_time = Histogram('order_execution_seconds', 'Order execution time')
order_execution_failures = Counter('order_execution_failures_total', 'Order execution failures')
order_success_rate = Gauge('order_success_rate', 'Order success rate percentage')

model_predictions = Counter('model_predictions_total', 'Total model predictions', ['model_type', 'signal'])
model_accuracy = Gauge('model_accuracy', 'Model prediction accuracy', ['model_type'])
model_drift_score = Gauge('model_drift_score', 'Model drift detection score', ['model_type'])

risk_alerts = Counter('risk_alerts_total', 'Risk alerts triggered', ['alert_type', 'severity'])
system_health = Gauge('system_health_score', 'Overall system health score')

# Service metrics
service_response_time = Histogram('service_response_seconds', 'Service response time', ['service'])
service_requests = Counter('service_requests_total', 'Service requests', ['service', 'method', 'status'])

class TradingMetricsExporter:
    """Exports trading-specific metrics to Prometheus."""
    
    def __init__(self, db_url: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.db_url = db_url
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.db_pool = None
        
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def collect_portfolio_metrics(self):
        """Collect portfolio-related metrics."""
        try:
            async with self.db_pool.acquire() as conn:
                # Portfolio value
                result = await conn.fetchrow("""
                    SELECT 
                        SUM(quantity * current_price) as total_value,
                        COUNT(*) as position_count,
                        SUM(unrealized_pnl) as daily_pnl
                    FROM positions 
                    WHERE is_active = true
                """)
                
                if result:
                    portfolio_value.set(result['total_value'] or 0)
                    portfolio_positions.set(result['position_count'] or 0)
                    portfolio_pnl.set(result['daily_pnl'] or 0)
                
                # VaR calculation (simplified)
                var_result = await conn.fetchrow("""
                    SELECT var_1d 
                    FROM risk_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                if var_result:
                    portfolio_var_1d.set(var_result['var_1d'] or 0)
                
                # Trading volume (last 24h)
                volume_result = await conn.fetchrow("""
                    SELECT SUM(quantity * price) as volume
                    FROM trades 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                if volume_result:
                    trading_volume.set(volume_result['volume'] or 0)
                
        except Exception as e:
            logger.error(f"Error collecting portfolio metrics: {e}")
    
    async def collect_order_metrics(self):
        """Collect order execution metrics."""
        try:
            async with self.db_pool.acquire() as conn:
                # Order success rate (last hour)
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_orders,
                        COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as successful_orders,
                        COUNT(CASE WHEN status = 'REJECTED' THEN 1 END) as failed_orders,
                        AVG(EXTRACT(EPOCH FROM (filled_at - created_at))) as avg_execution_time
                    FROM orders 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                if result and result['total_orders'] > 0:
                    success_rate = (result['successful_orders'] / result['total_orders']) * 100
                    order_success_rate.set(success_rate)
                    
                    if result['failed_orders']:
                        order_execution_failures.inc(result['failed_orders'])
                    
                    if result['avg_execution_time']:
                        order_execution_time.observe(result['avg_execution_time'])
                
        except Exception as e:
            logger.error(f"Error collecting order metrics: {e}")
    
    async def collect_model_metrics(self):
        """Collect ML model metrics."""
        try:
            # Get model metrics from Redis cache
            model_stats = self.redis_client.get('model_stats')
            if model_stats:
                stats = json.loads(model_stats)
                
                for model_type, metrics in stats.items():
                    model_accuracy.labels(model_type=model_type).set(metrics.get('accuracy', 0))
                    model_drift_score.labels(model_type=model_type).set(metrics.get('drift_score', 0))
                    
                    # Prediction counts
                    for signal, count in metrics.get('predictions', {}).items():
                        model_predictions.labels(model_type=model_type, signal=signal).inc(count)
            
            # Get drift alerts
            drift_alerts = self.redis_client.get('drift_alerts')
            if drift_alerts:
                alerts = json.loads(drift_alerts)
                for alert in alerts:
                    risk_alerts.labels(
                        alert_type='drift',
                        severity=alert.get('severity', 'unknown')
                    ).inc()
                
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
    
    async def collect_system_health_metrics(self):
        """Collect system health metrics."""
        try:
            # Calculate overall system health score
            health_score = 100.0
            
            # Check service health
            services = ['decision-service', 'risk-service', 'execution-service']
            healthy_services = 0
            
            for service in services:
                health_status = self.redis_client.get(f'health:{service}')
                if health_status == 'healthy':
                    healthy_services += 1
            
            service_health_score = (healthy_services / len(services)) * 100
            
            # Check database health
            db_health_score = 100.0
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
            except:
                db_health_score = 0.0
            
            # Check Redis health
            redis_health_score = 100.0
            try:
                self.redis_client.ping()
            except:
                redis_health_score = 0.0
            
            # Calculate weighted health score
            health_score = (
                service_health_score * 0.5 +
                db_health_score * 0.3 +
                redis_health_score * 0.2
            )
            
            system_health.set(health_score)
            
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {e}")
    
    async def collect_all_metrics(self):
        """Collect all metrics."""
        try:
            await asyncio.gather(
                self.collect_portfolio_metrics(),
                self.collect_order_metrics(),
                self.collect_model_metrics(),
                self.collect_system_health_metrics()
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def run(self):
        """Main metrics collection loop."""
        await self.initialize()
        
        logger.info("Starting trading metrics exporter on port 8090")
        start_http_server(8090)
        
        while True:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    import os
    
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/trading')
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))
    
    exporter = TradingMetricsExporter(db_url, redis_host, redis_port)
    asyncio.run(exporter.run())
