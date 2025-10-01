-- Optimized Indexes for Trading Ecosystem
-- These indexes are designed for high-frequency time series queries

-- =============================================================================
-- MARKET DATA INDEXES
-- =============================================================================

-- Primary indexes for market data queries
CREATE INDEX CONCURRENTLY idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX CONCURRENTLY idx_market_data_exchange_symbol ON market_data (exchange, symbol);
CREATE INDEX CONCURRENTLY idx_market_data_time_symbol ON market_data (time DESC, symbol);

-- Composite index for OHLCV queries
CREATE INDEX CONCURRENTLY idx_market_data_composite ON market_data (symbol, exchange, time DESC);

-- Order book indexes
CREATE INDEX CONCURRENTLY idx_order_book_symbol_time ON order_book (symbol, time DESC);
CREATE INDEX CONCURRENTLY idx_order_book_symbol_side_time ON order_book (symbol, side, time DESC);

-- Trades indexes
CREATE INDEX CONCURRENTLY idx_trades_symbol_time ON trades (symbol, time DESC);
CREATE INDEX CONCURRENTLY idx_trades_exchange_symbol ON trades (exchange, symbol);

-- =============================================================================
-- ALTERNATIVE DATA INDEXES
-- =============================================================================

-- News sentiment indexes
CREATE INDEX CONCURRENTLY idx_news_sentiment_time ON news_sentiment (time DESC);
CREATE INDEX CONCURRENTLY idx_news_sentiment_symbol_time ON news_sentiment (symbol, time DESC);
CREATE INDEX CONCURRENTLY idx_news_sentiment_source ON news_sentiment (source);
CREATE INDEX CONCURRENTLY idx_news_sentiment_score ON news_sentiment (sentiment_score);

-- On-chain data indexes
CREATE INDEX CONCURRENTLY idx_onchain_data_symbol_metric ON onchain_data (symbol, metric_name, time DESC);
CREATE INDEX CONCURRENTLY idx_onchain_data_time ON onchain_data (time DESC);

-- =============================================================================
-- TRADING SYSTEM INDEXES
-- =============================================================================

-- Strategy indexes
CREATE INDEX CONCURRENTLY idx_strategies_active ON strategies (is_active) WHERE is_active = true;
CREATE INDEX CONCURRENTLY idx_strategies_name ON strategies (name);

-- Position indexes
CREATE INDEX CONCURRENTLY idx_positions_strategy_id ON positions (strategy_id);
CREATE INDEX CONCURRENTLY idx_positions_symbol ON positions (symbol);
CREATE INDEX CONCURRENTLY idx_positions_status ON positions (status);
CREATE INDEX CONCURRENTLY idx_positions_strategy_symbol ON positions (strategy_id, symbol);
CREATE INDEX CONCURRENTLY idx_positions_opened_at ON positions (opened_at DESC);

-- Order indexes
CREATE INDEX CONCURRENTLY idx_orders_strategy_id ON orders (strategy_id);
CREATE INDEX CONCURRENTLY idx_orders_position_id ON orders (position_id);
CREATE INDEX CONCURRENTLY idx_orders_symbol ON orders (symbol);
CREATE INDEX CONCURRENTLY idx_orders_status ON orders (status);
CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders (created_at DESC);
CREATE INDEX CONCURRENTLY idx_orders_exchange_order_id ON orders (exchange_order_id);

-- Portfolio performance indexes
CREATE INDEX CONCURRENTLY idx_portfolio_performance_strategy_time ON portfolio_performance (strategy_id, time DESC);
CREATE INDEX CONCURRENTLY idx_portfolio_performance_time ON portfolio_performance (time DESC);

-- =============================================================================
-- RISK MANAGEMENT INDEXES
-- =============================================================================

-- Risk metrics indexes
CREATE INDEX CONCURRENTLY idx_risk_metrics_strategy_time ON risk_metrics (strategy_id, time DESC);
CREATE INDEX CONCURRENTLY idx_risk_metrics_metric_name ON risk_metrics (metric_name);
CREATE INDEX CONCURRENTLY idx_risk_metrics_alert_level ON risk_metrics (alert_level);

-- Circuit breaker events indexes
CREATE INDEX CONCURRENTLY idx_circuit_breaker_strategy ON circuit_breaker_events (strategy_id);
CREATE INDEX CONCURRENTLY idx_circuit_breaker_time ON circuit_breaker_events (time DESC);
CREATE INDEX CONCURRENTLY idx_circuit_breaker_trigger_type ON circuit_breaker_events (trigger_type);

-- =============================================================================
-- MLOPS INDEXES
-- =============================================================================

-- Model training runs indexes
CREATE INDEX CONCURRENTLY idx_model_training_runs_run_id ON model_training_runs (run_id);
CREATE INDEX CONCURRENTLY idx_model_training_runs_model_name ON model_training_runs (model_name);
CREATE INDEX CONCURRENTLY idx_model_training_runs_status ON model_training_runs (status);
CREATE INDEX CONCURRENTLY idx_model_training_runs_started_at ON model_training_runs (started_at DESC);

-- Model deployments indexes
CREATE INDEX CONCURRENTLY idx_model_deployments_strategy_id ON model_deployments (strategy_id);
CREATE INDEX CONCURRENTLY idx_model_deployments_model_name ON model_deployments (model_name);
CREATE INDEX CONCURRENTLY idx_model_deployments_status ON model_deployments (status);
CREATE INDEX CONCURRENTLY idx_model_deployments_deployed_at ON model_deployments (deployed_at DESC);

-- Feature drift indexes
CREATE INDEX CONCURRENTLY idx_feature_drift_model_feature ON feature_drift (model_name, feature_name, time DESC);
CREATE INDEX CONCURRENTLY idx_feature_drift_detected ON feature_drift (is_drift_detected, time DESC);

-- =============================================================================
-- SYSTEM MONITORING INDEXES
-- =============================================================================

-- System logs indexes
CREATE INDEX CONCURRENTLY idx_system_logs_service_time ON system_logs (service_name, time DESC);
CREATE INDEX CONCURRENTLY idx_system_logs_level ON system_logs (log_level);
CREATE INDEX CONCURRENTLY idx_system_logs_trace_id ON system_logs (trace_id);
CREATE INDEX CONCURRENTLY idx_system_logs_time ON system_logs (time DESC);

-- System metrics indexes
CREATE INDEX CONCURRENTLY idx_system_metrics_service_metric ON system_metrics (service_name, metric_name, time DESC);
CREATE INDEX CONCURRENTLY idx_system_metrics_time ON system_metrics (time DESC);

-- =============================================================================
-- USER MANAGEMENT INDEXES
-- =============================================================================

-- Users indexes
CREATE INDEX CONCURRENTLY idx_users_username ON users (username);
CREATE INDEX CONCURRENTLY idx_users_email ON users (email);
CREATE INDEX CONCURRENTLY idx_users_role ON users (role);
CREATE INDEX CONCURRENTLY idx_users_active ON users (is_active) WHERE is_active = true;

-- User sessions indexes
CREATE INDEX CONCURRENTLY idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX CONCURRENTLY idx_user_sessions_token ON user_sessions (session_token);
CREATE INDEX CONCURRENTLY idx_user_sessions_expires_at ON user_sessions (expires_at);

-- =============================================================================
-- PARTIAL INDEXES FOR PERFORMANCE
-- =============================================================================

-- Only index active positions
CREATE INDEX CONCURRENTLY idx_positions_active ON positions (strategy_id, symbol) WHERE status = 'open';

-- Only index pending/active orders
CREATE INDEX CONCURRENTLY idx_orders_active ON orders (strategy_id, symbol) WHERE status IN ('pending', 'partial');

-- Only index recent logs (last 30 days)
CREATE INDEX CONCURRENTLY idx_system_logs_recent ON system_logs (service_name, time DESC) 
WHERE time > NOW() - INTERVAL '30 days';

-- Only index drift detected events
CREATE INDEX CONCURRENTLY idx_feature_drift_alerts ON feature_drift (model_name, time DESC) 
WHERE is_drift_detected = true;

-- =============================================================================
-- HYPERTABLE COMPRESSION POLICIES
-- =============================================================================

-- Enable compression on older data to save space
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_compression_policy('order_book', INTERVAL '1 day');
SELECT add_compression_policy('trades', INTERVAL '7 days');
SELECT add_compression_policy('news_sentiment', INTERVAL '30 days');
SELECT add_compression_policy('onchain_data', INTERVAL '30 days');
SELECT add_compression_policy('portfolio_performance', INTERVAL '30 days');
SELECT add_compression_policy('risk_metrics', INTERVAL '30 days');
SELECT add_compression_policy('feature_drift', INTERVAL '30 days');
SELECT add_compression_policy('system_logs', INTERVAL '7 days');
SELECT add_compression_policy('system_metrics', INTERVAL '7 days');

-- =============================================================================
-- DATA RETENTION POLICIES
-- =============================================================================

-- Automatically drop old data to manage storage
SELECT add_retention_policy('order_book', INTERVAL '30 days');
SELECT add_retention_policy('system_logs', INTERVAL '90 days');
SELECT add_retention_policy('system_metrics', INTERVAL '1 year');
SELECT add_retention_policy('feature_drift', INTERVAL '1 year');

-- Keep market data and trades for longer periods
SELECT add_retention_policy('market_data', INTERVAL '5 years');
SELECT add_retention_policy('trades', INTERVAL '5 years');
