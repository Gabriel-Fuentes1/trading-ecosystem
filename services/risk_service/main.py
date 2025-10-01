"""
Risk Management Service
Microservice for portfolio risk management and position sizing.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis
import json
from contextlib import asynccontextmanager

# Risk management libraries
from scipy import stats
import quantlib as ql

# Custom imports
from models.risk_models import VaRCalculator, PortfolioRiskAnalyzer
from utils.database import DatabaseManager
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()

# Redis client
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

# Global variables
var_calculator = None
risk_analyzer = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global var_calculator, risk_analyzer, db_manager
    
    try:
        # Initialize components
        var_calculator = VaRCalculator()
        risk_analyzer = PortfolioRiskAnalyzer()
        db_manager = DatabaseManager(settings.DATABASE_URL)
        
        logger.info("Risk service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize risk service: {e}")
        raise
    finally:
        # Cleanup
        if db_manager:
            await db_manager.close()
        logger.info("Risk service shutting down")

app = FastAPI(
    title="Trading Risk Service",
    description="Microservice for risk management and position sizing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Position(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # 'LONG' or 'SHORT'
    entry_timestamp: datetime

class PortfolioRiskRequest(BaseModel):
    positions: List[Position]
    portfolio_value: float = Field(..., gt=0)
    confidence_level: float = Field(default=0.95, ge=0.9, le=0.99)
    time_horizon: int = Field(default=1, ge=1, le=30)  # days

class RiskMetrics(BaseModel):
    var_1d: float  # 1-day Value at Risk
    var_10d: float  # 10-day Value at Risk
    expected_shortfall: float  # Conditional VaR
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    volatility: float
    correlation_matrix: Dict[str, Dict[str, float]]

class PositionSizingRequest(BaseModel):
    symbol: str
    signal_strength: float = Field(..., ge=0.0, le=1.0)
    predicted_return: float
    predicted_volatility: float
    portfolio_value: float = Field(..., gt=0)
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)  # 2% default
    max_position_size: float = Field(default=0.2, ge=0.01, le=0.5)  # 20% max

class PositionSizingResponse(BaseModel):
    recommended_quantity: float
    position_value: float
    position_percentage: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    kelly_fraction: float

class RiskAlert(BaseModel):
    alert_type: str  # 'VAR_BREACH', 'CONCENTRATION', 'CORRELATION', 'DRAWDOWN'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    affected_positions: List[str]
    recommended_action: str
    timestamp: datetime

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check database connection
        await db_manager.test_connection()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "service": "risk_service",
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/calculate_portfolio_risk", response_model=RiskMetrics)
async def calculate_portfolio_risk(request: PortfolioRiskRequest):
    """Calculate comprehensive portfolio risk metrics."""
    try:
        logger.info(f"Calculating risk for portfolio with {len(request.positions)} positions")
        
        if not request.positions:
            raise HTTPException(status_code=400, detail="No positions provided")
        
        # Get historical data for all symbols
        symbols = [pos.symbol for pos in request.positions]
        historical_data = await db_manager.get_historical_data(
            symbols=symbols,
            days=252  # 1 year of data
        )
        
        # Calculate returns
        returns_data = {}
        for symbol in symbols:
            if symbol in historical_data:
                prices = historical_data[symbol]['close']
                returns = prices.pct_change().dropna()
                returns_data[symbol] = returns
        
        # Calculate position weights
        total_value = sum(pos.quantity * pos.current_price for pos in request.positions)
        weights = {}
        for pos in request.positions:
            position_value = pos.quantity * pos.current_price
            weights[pos.symbol] = position_value / total_value
        
        # Calculate VaR
        var_1d = var_calculator.calculate_portfolio_var(
            returns_data=returns_data,
            weights=weights,
            confidence_level=request.confidence_level,
            time_horizon=1
        )
        
        var_10d = var_calculator.calculate_portfolio_var(
            returns_data=returns_data,
            weights=weights,
            confidence_level=request.confidence_level,
            time_horizon=10
        )
        
        # Calculate Expected Shortfall (Conditional VaR)
        expected_shortfall = var_calculator.calculate_expected_shortfall(
            returns_data=returns_data,
            weights=weights,
            confidence_level=request.confidence_level
        )
        
        # Calculate other risk metrics
        portfolio_returns = risk_analyzer.calculate_portfolio_returns(returns_data, weights)
        
        risk_metrics = RiskMetrics(
            var_1d=var_1d * request.portfolio_value,
            var_10d=var_10d * request.portfolio_value,
            expected_shortfall=expected_shortfall * request.portfolio_value,
            maximum_drawdown=risk_analyzer.calculate_max_drawdown(portfolio_returns),
            sharpe_ratio=risk_analyzer.calculate_sharpe_ratio(portfolio_returns),
            sortino_ratio=risk_analyzer.calculate_sortino_ratio(portfolio_returns),
            beta=risk_analyzer.calculate_beta(portfolio_returns),
            volatility=portfolio_returns.std() * np.sqrt(252),  # Annualized
            correlation_matrix=risk_analyzer.calculate_correlation_matrix(returns_data)
        )
        
        logger.info(f"Portfolio VaR (1d): ${risk_metrics.var_1d:,.2f}, Sharpe: {risk_metrics.sharpe_ratio:.3f}")
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_position_size", response_model=PositionSizingResponse)
async def calculate_position_size(request: PositionSizingRequest):
    """Calculate optimal position size based on risk parameters."""
    try:
        logger.info(f"Calculating position size for {request.symbol}")
        
        # Get current price
        current_price = await db_manager.get_current_price(request.symbol)
        if not current_price:
            raise HTTPException(status_code=404, detail=f"Price not found for {request.symbol}")
        
        # Calculate Kelly fraction
        win_rate = 0.5 + (request.signal_strength - 0.5) * 0.4  # Adjust based on signal strength
        avg_win = abs(request.predicted_return)
        avg_loss = request.predicted_volatility
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Calculate position size based on risk per trade
        risk_amount = request.portfolio_value * request.risk_per_trade
        
        # Stop loss calculation (2 * volatility)
        stop_loss_distance = 2 * request.predicted_volatility * current_price
        
        if request.predicted_return > 0:  # Long position
            stop_loss_price = current_price - stop_loss_distance
            take_profit_price = current_price * (1 + request.predicted_return)
        else:  # Short position
            stop_loss_price = current_price + stop_loss_distance
            take_profit_price = current_price * (1 + request.predicted_return)
        
        # Position size based on risk
        position_size_by_risk = risk_amount / stop_loss_distance
        
        # Position size based on Kelly criterion
        position_size_by_kelly = (request.portfolio_value * kelly_fraction) / current_price
        
        # Take the minimum of the two approaches
        recommended_quantity = min(position_size_by_risk, position_size_by_kelly)
        
        # Apply maximum position size constraint
        max_quantity = (request.portfolio_value * request.max_position_size) / current_price
        recommended_quantity = min(recommended_quantity, max_quantity)
        
        # Ensure minimum viable position
        recommended_quantity = max(recommended_quantity, 0.001)
        
        position_value = recommended_quantity * current_price
        position_percentage = position_value / request.portfolio_value
        
        # Risk-reward ratio
        risk_per_share = abs(current_price - stop_loss_price)
        reward_per_share = abs(take_profit_price - current_price)
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        response = PositionSizingResponse(
            recommended_quantity=recommended_quantity,
            position_value=position_value,
            position_percentage=position_percentage,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_reward_ratio=risk_reward_ratio,
            kelly_fraction=kelly_fraction
        )
        
        logger.info(f"Position size for {request.symbol}: {recommended_quantity:.6f} ({position_percentage:.2%})")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating position size for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_risk_alerts", response_model=List[RiskAlert])
async def check_risk_alerts(request: PortfolioRiskRequest):
    """Check for risk alerts and violations."""
    try:
        logger.info("Checking for risk alerts")
        
        alerts = []
        
        # Calculate current risk metrics
        risk_metrics = await calculate_portfolio_risk(request)
        
        # VaR breach alert
        var_limit = request.portfolio_value * 0.05  # 5% of portfolio
        if risk_metrics.var_1d > var_limit:
            alerts.append(RiskAlert(
                alert_type="VAR_BREACH",
                severity="HIGH" if risk_metrics.var_1d > var_limit * 1.5 else "MEDIUM",
                message=f"1-day VaR (${risk_metrics.var_1d:,.2f}) exceeds limit (${var_limit:,.2f})",
                affected_positions=[pos.symbol for pos in request.positions],
                recommended_action="Reduce position sizes or hedge exposure",
                timestamp=datetime.now()
            ))
        
        # Concentration risk alert
        position_weights = {}
        total_value = sum(pos.quantity * pos.current_price for pos in request.positions)
        
        for pos in request.positions:
            position_value = pos.quantity * pos.current_price
            weight = position_value / total_value
            position_weights[pos.symbol] = weight
            
            # Check for over-concentration (>30% in single position)
            if weight > 0.3:
                alerts.append(RiskAlert(
                    alert_type="CONCENTRATION",
                    severity="HIGH",
                    message=f"{pos.symbol} represents {weight:.1%} of portfolio (>30%)",
                    affected_positions=[pos.symbol],
                    recommended_action="Diversify portfolio by reducing position size",
                    timestamp=datetime.now()
                ))
        
        # Maximum drawdown alert
        if risk_metrics.maximum_drawdown > 0.15:  # 15% drawdown
            alerts.append(RiskAlert(
                alert_type="DRAWDOWN",
                severity="CRITICAL" if risk_metrics.maximum_drawdown > 0.25 else "HIGH",
                message=f"Maximum drawdown is {risk_metrics.maximum_drawdown:.1%}",
                affected_positions=[pos.symbol for pos in request.positions],
                recommended_action="Review strategy and consider reducing risk",
                timestamp=datetime.now()
            ))
        
        # Low Sharpe ratio alert
        if risk_metrics.sharpe_ratio < 0.5:
            alerts.append(RiskAlert(
                alert_type="PERFORMANCE",
                severity="MEDIUM",
                message=f"Sharpe ratio is low ({risk_metrics.sharpe_ratio:.2f})",
                affected_positions=[pos.symbol for pos in request.positions],
                recommended_action="Review strategy performance and risk-adjusted returns",
                timestamp=datetime.now()
            ))
        
        logger.info(f"Generated {len(alerts)} risk alerts")
        return alerts
        
    except Exception as e:
        logger.error(f"Error checking risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk_limits")
async def get_risk_limits():
    """Get current risk limits and constraints."""
    return {
        "max_var_percentage": 5.0,  # 5% of portfolio
        "max_position_concentration": 30.0,  # 30% max in single position
        "max_sector_concentration": 40.0,  # 40% max in single sector
        "max_drawdown_threshold": 15.0,  # 15% maximum drawdown
        "min_sharpe_ratio": 0.5,
        "max_leverage": 2.0,
        "risk_per_trade": 2.0,  # 2% risk per trade
        "correlation_threshold": 0.8  # High correlation warning
    }

@app.put("/update_risk_limits")
async def update_risk_limits(limits: Dict[str, float]):
    """Update risk limits (admin only)."""
    try:
        # Store in Redis for quick access
        redis_client.hset("risk_limits", mapping=limits)
        
        logger.info("Risk limits updated")
        return {"message": "Risk limits updated successfully", "limits": limits}
        
    except Exception as e:
        logger.error(f"Error updating risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
