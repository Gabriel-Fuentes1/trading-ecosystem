-- Trading Ecosystem Database Schema
-- TimescaleDB optimized for time series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Market Data Table (Time Series)
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    vwap DECIMAL(20,8),
    trades INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time series optimization
SELECT create_hypertable('market_data', 'time');

-- Order Book Data (Level 2)
CREATE TABLE order_book (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'bid' or 'ask'
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    level INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('order_book', 'time');

-- Trades Table
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'buy' or 'sell'
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    trade_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('trades', 'time');

-- =============================================================================
-- ALTERNATIVE DATA TABLES
-- =============================================================================

-- News and Sentiment Data
CREATE TABLE news_sentiment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20),
    title TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100) NOT NULL,
    sentiment_score DECIMAL(5,4), -- -1 to 1
    sentiment_label VARCHAR(20), -- 'positive', 'negative', 'neutral'
    confidence DECIMAL(5,4), -- 0 to 1
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('news_sentiment', 'time');

-- On-Chain Data (for crypto assets)
CREATE TABLE onchain_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(30,8) NOT NULL,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('onchain_data', 'time');

-- =============================================================================
-- TRADING SYSTEM TABLES
-- =============================================================================

-- Strategies Table
CREATE TABLE strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    model_path VARCHAR(500),
    model_version VARCHAR(50),
    parameters JSONB,
    is_active BOOLEAN DEFAULT false,
    risk_limits JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Positions Table
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'long' or 'short'
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'partial'
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders Table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id),
    position_id UUID REFERENCES positions(id),
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'buy' or 'sell'
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', etc.
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_fill_price DECIMAL(20,8),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'cancelled', 'rejected'
    exchange_order_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ
);

-- Portfolio Performance Table
CREATE TABLE portfolio_performance (
    time TIMESTAMPTZ NOT NULL,
    strategy_id UUID REFERENCES strategies(id),
    total_value DECIMAL(20,8) NOT NULL,
    cash_balance DECIMAL(20,8) NOT NULL,
    unrealized_pnl DECIMAL(20,8) NOT NULL,
    realized_pnl DECIMAL(20,8) NOT NULL,
    drawdown DECIMAL(10,6) NOT NULL,
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    var_95 DECIMAL(20,8), -- Value at Risk 95%
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('portfolio_performance', 'time');

-- =============================================================================
-- RISK MANAGEMENT TABLES
-- =============================================================================

-- Risk Metrics Table
CREATE TABLE risk_metrics (
    time TIMESTAMPTZ NOT NULL,
    strategy_id UUID REFERENCES strategies(id),
    symbol VARCHAR(20),
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    threshold_value DECIMAL(20,8),
    alert_level VARCHAR(20), -- 'info', 'warning', 'critical'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('risk_metrics', 'time');

-- Circuit Breaker Events
CREATE TABLE circuit_breaker_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    strategy_id UUID REFERENCES strategies(id),
    trigger_type VARCHAR(50) NOT NULL, -- 'drawdown', 'var_breach', 'position_limit'
    trigger_value DECIMAL(20,8) NOT NULL,
    threshold_value DECIMAL(20,8) NOT NULL,
    action_taken VARCHAR(100) NOT NULL,
    description TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- MLOPS TABLES
-- =============================================================================

-- Model Training Runs
CREATE TABLE model_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(100) NOT NULL UNIQUE, -- MLflow run ID
    experiment_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    parameters JSONB,
    metrics JSONB,
    artifacts_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model Deployment History
CREATE TABLE model_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID REFERENCES strategies(id),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    deployment_type VARCHAR(20) DEFAULT 'production', -- 'staging', 'production'
    deployed_at TIMESTAMPTZ NOT NULL,
    deployed_by VARCHAR(100),
    rollback_version VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'inactive', 'rollback'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature Drift Monitoring
CREATE TABLE feature_drift (
    time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    drift_score DECIMAL(10,6) NOT NULL,
    threshold DECIMAL(10,6) NOT NULL,
    is_drift_detected BOOLEAN NOT NULL,
    statistical_test VARCHAR(50), -- 'ks_test', 'chi2_test', etc.
    p_value DECIMAL(10,8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('feature_drift', 'time');

-- =============================================================================
-- SYSTEM MONITORING TABLES
-- =============================================================================

-- System Logs
CREATE TABLE system_logs (
    time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    log_level VARCHAR(10) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message TEXT NOT NULL,
    component VARCHAR(100),
    trace_id VARCHAR(100),
    span_id VARCHAR(100),
    user_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('system_logs', 'time');

-- System Metrics
CREATE TABLE system_metrics (
    time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    unit VARCHAR(20),
    tags JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('system_metrics', 'time');

-- =============================================================================
-- USER MANAGEMENT TABLES
-- =============================================================================

-- Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'viewer', -- 'admin', 'trader', 'viewer'
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User Sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TRIGGERS FOR UPDATED_AT
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to relevant tables
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, password_hash, role) VALUES 
('admin', 'admin@trading-system.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBdXzgVrqUm/PG', 'admin');

-- Insert default strategy
INSERT INTO strategies (name, description, parameters, risk_limits) VALUES 
('Default RL Strategy', 'Reinforcement Learning based trading strategy', 
 '{"lookback_window": 100, "action_space": 3, "reward_function": "sharpe"}',
 '{"max_position_size": 0.1, "max_drawdown": 0.05, "var_limit": 0.02}');
