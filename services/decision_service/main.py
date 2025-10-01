"""
Decision Service - Trading Signal Generation
Microservice for generating trading decisions using ML models.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis
import json
import joblib
from contextlib import asynccontextmanager

# Circuit breaker
from pybreaker import CircuitBreaker

# Custom imports
from models.trading_models import TradingModelPredictor
from utils.data_fetcher import MarketDataFetcher
from utils.feature_engineering import FeatureEngineer
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()

# Redis client
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

# Circuit breaker for external services
model_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
data_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)

# Global variables
model_predictor = None
data_fetcher = None
feature_engineer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global model_predictor, data_fetcher, feature_engineer
    
    try:
        # Initialize components
        model_predictor = TradingModelPredictor(settings.MODEL_PATH)
        data_fetcher = MarketDataFetcher(settings.DATABASE_URL)
        feature_engineer = FeatureEngineer()
        
        logger.info("Decision service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize decision service: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Decision service shutting down")

app = FastAPI(
    title="Trading Decision Service",
    description="Microservice for generating trading decisions",
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
class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default="1h", description="Timeframe for analysis")
    lookback_periods: int = Field(default=100, description="Number of periods to analyze")
    model_type: str = Field(default="ensemble", description="Model type to use")

class TradingSignal(BaseModel):
    symbol: str
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_return: float
    risk_score: float = Field(..., ge=0.0, le=1.0)
    features: Dict[str, float]
    model_version: str
    explanation: Dict[str, Any]

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str]
    investment_amount: float = Field(..., gt=0)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    max_positions: int = Field(default=10, ge=1, le=50)

class PortfolioAllocation(BaseModel):
    symbol: str
    allocation_percentage: float
    allocation_amount: float
    expected_return: float
    risk_contribution: float

class PortfolioOptimizationResponse(BaseModel):
    allocations: List[PortfolioAllocation]
    total_expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    optimization_timestamp: datetime

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check model availability
        if model_predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "service": "decision_service",
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/generate_signal", response_model=TradingSignal)
async def generate_trading_signal(request: TradingSignalRequest):
    """Generate trading signal for a given symbol."""
    try:
        logger.info(f"Generating signal for {request.symbol}")
        
        # Check cache first
        cache_key = f"signal:{request.symbol}:{request.timeframe}:{request.model_type}"
        cached_signal = redis_client.get(cache_key)
        
        if cached_signal:
            logger.info(f"Returning cached signal for {request.symbol}")
            return TradingSignal(**json.loads(cached_signal))
        
        # Fetch market data with circuit breaker
        @data_breaker
        def fetch_data():
            return data_fetcher.get_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.lookback_periods
            )
        
        market_data = fetch_data()
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Engineer features
        features = feature_engineer.create_features(market_data)
        
        # Generate prediction with circuit breaker
        @model_breaker
        def predict():
            return model_predictor.predict(
                features=features,
                model_type=request.model_type
            )
        
        prediction_result = predict()
        
        # Create trading signal
        signal = TradingSignal(
            symbol=request.symbol,
            timestamp=datetime.now(),
            signal=prediction_result['signal'],
            confidence=prediction_result['confidence'],
            predicted_return=prediction_result['predicted_return'],
            risk_score=prediction_result['risk_score'],
            features=prediction_result['features'],
            model_version=prediction_result['model_version'],
            explanation=prediction_result['explanation']
        )
        
        # Cache the signal (expire in 5 minutes)
        redis_client.setex(
            cache_key,
            300,
            signal.model_dump_json()
        )
        
        logger.info(f"Generated {signal.signal} signal for {request.symbol} with confidence {signal.confidence:.3f}")
        return signal
        
    except Exception as e:
        logger.error(f"Error generating signal for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize_portfolio", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio allocation using modern portfolio theory."""
    try:
        logger.info(f"Optimizing portfolio for {len(request.symbols)} symbols")
        
        # Generate signals for all symbols
        signals = []
        for symbol in request.symbols:
            signal_request = TradingSignalRequest(symbol=symbol)
            signal = await generate_trading_signal(signal_request)
            signals.append(signal)
        
        # Filter signals with positive expected returns and reasonable confidence
        viable_signals = [
            s for s in signals 
            if s.predicted_return > 0 and s.confidence > 0.6 and s.signal in ['BUY', 'HOLD']
        ]
        
        if not viable_signals:
            raise HTTPException(status_code=400, detail="No viable investment opportunities found")
        
        # Limit to max positions
        viable_signals = sorted(viable_signals, key=lambda x: x.confidence * x.predicted_return, reverse=True)
        viable_signals = viable_signals[:request.max_positions]
        
        # Simple portfolio optimization (equal risk contribution)
        total_risk_adjusted_return = sum(s.predicted_return / s.risk_score for s in viable_signals)
        
        allocations = []
        total_expected_return = 0
        total_risk = 0
        
        for signal in viable_signals:
            # Risk-adjusted allocation
            risk_adjusted_return = signal.predicted_return / signal.risk_score
            allocation_percentage = (risk_adjusted_return / total_risk_adjusted_return) * (1 - request.risk_tolerance + 0.1)
            allocation_amount = request.investment_amount * allocation_percentage
            
            allocation = PortfolioAllocation(
                symbol=signal.symbol,
                allocation_percentage=allocation_percentage,
                allocation_amount=allocation_amount,
                expected_return=signal.predicted_return,
                risk_contribution=signal.risk_score * allocation_percentage
            )
            allocations.append(allocation)
            
            total_expected_return += signal.predicted_return * allocation_percentage
            total_risk += (signal.risk_score * allocation_percentage) ** 2
        
        # Calculate portfolio metrics
        portfolio_risk = np.sqrt(total_risk)
        sharpe_ratio = total_expected_return / portfolio_risk if portfolio_risk > 0 else 0
        
        response = PortfolioOptimizationResponse(
            allocations=allocations,
            total_expected_return=total_expected_return,
            portfolio_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            optimization_timestamp=datetime.now()
        )
        
        logger.info(f"Portfolio optimized: {len(allocations)} positions, Sharpe ratio: {sharpe_ratio:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def get_model_status():
    """Get current model status and performance metrics."""
    try:
        if model_predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        status = model_predictor.get_status()
        return {
            "model_loaded": True,
            "model_version": status.get('version', 'unknown'),
            "last_updated": status.get('last_updated'),
            "performance_metrics": status.get('metrics', {}),
            "feature_importance": status.get('feature_importance', {}),
            "prediction_count": status.get('prediction_count', 0)
        }
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain_model")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining in the background."""
    try:
        def retrain_model():
            logger.info("Starting model retraining...")
            # This would trigger the MLOps pipeline
            # For now, just log the action
            logger.info("Model retraining triggered")
        
        background_tasks.add_task(retrain_model)
        
        return {
            "message": "Model retraining triggered",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
