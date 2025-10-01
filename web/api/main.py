"""
Trading Platform API Gateway
Main FastAPI application serving the web interface and API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import logging
import json
import httpx
from datetime import datetime, timedelta
import redis
from contextlib import asynccontextmanager
import os

# Authentication
import jwt
from passlib.context import CryptContext

# Custom imports
from config.settings import get_settings
from models.user_models import User, UserCreate, UserLogin
from utils.auth import AuthManager
from utils.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()

# Global variables
auth_manager = None
db_manager = None
redis_client = None
websocket_connections = []

# HTTP clients for microservices
decision_client = httpx.AsyncClient(base_url="http://decision-service:8001")
risk_client = httpx.AsyncClient(base_url="http://risk-service:8002")
execution_client = httpx.AsyncClient(base_url="http://execution-service:8003")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global auth_manager, db_manager, redis_client
    
    try:
        # Initialize components
        auth_manager = AuthManager(settings.JWT_SECRET_KEY)
        db_manager = DatabaseManager(settings.DATABASE_URL)
        redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
        
        # Start background tasks
        asyncio.create_task(broadcast_market_data())
        
        logger.info("API Gateway initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize API Gateway: {e}")
        raise
    finally:
        # Cleanup
        await decision_client.aclose()
        await risk_client.aclose()
        await execution_client.aclose()
        if db_manager:
            await db_manager.close()
        logger.info("API Gateway shutting down")

app = FastAPI(
    title="Trading Platform API",
    description="API Gateway for Quantitative Trading Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class DashboardData(BaseModel):
    portfolio_value: float
    daily_pnl: float
    daily_pnl_percentage: float
    total_positions: int
    active_orders: int
    risk_metrics: Dict[str, Any]
    top_performers: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percentage: float
    volume: float
    timestamp: datetime

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        payload = auth_manager.verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db_manager.get_user(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check microservices
        services_status = {}
        
        try:
            response = await decision_client.get("/health", timeout=5.0)
            services_status["decision_service"] = response.status_code == 200
        except:
            services_status["decision_service"] = False
        
        try:
            response = await risk_client.get("/health", timeout=5.0)
            services_status["risk_service"] = response.status_code == 200
        except:
            services_status["risk_service"] = False
        
        try:
            response = await execution_client.get("/health", timeout=5.0)
            services_status["execution_service"] = response.status_code == 200
        except:
            services_status["execution_service"] = False
        
        # Check Redis
        try:
            redis_client.ping()
            services_status["redis"] = True
        except:
            services_status["redis"] = False
        
        # Check database
        try:
            await db_manager.test_connection()
            services_status["database"] = True
        except:
            services_status["database"] = False
        
        all_healthy = all(services_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(),
            "services": services_status,
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# Authentication endpoints
@app.post("/auth/register")
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        # Check if user exists
        existing_user = await db_manager.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed_password = pwd_context.hash(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            is_active=True,
            created_at=datetime.now()
        )
        
        user_id = await db_manager.create_user(user)
        
        # Generate token
        token = auth_manager.create_access_token({"sub": str(user_id)})
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user.email,
                "username": user.username
            }
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login(user_data: UserLogin):
    """Login user."""
    try:
        # Get user
        user = await db_manager.get_user_by_email(user_data.email)
        if not user or not pwd_context.verify(user_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account disabled")
        
        # Generate token
        token = auth_manager.create_access_token({"sub": str(user.id)})
        
        # Update last login
        await db_manager.update_user_last_login(user.id)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username
            }
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard endpoints
@app.get("/api/dashboard", response_model=DashboardData)
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """Get dashboard data for the current user."""
    try:
        # Get portfolio data
        portfolio_data = await db_manager.get_user_portfolio(current_user.id)
        
        # Get positions
        positions_response = await execution_client.get("/positions")
        positions = positions_response.json() if positions_response.status_code == 200 else []
        
        # Get orders
        orders_response = await execution_client.get("/orders", params={"limit": 10})
        orders = orders_response.json() if orders_response.status_code == 200 else []
        
        # Calculate metrics
        portfolio_value = sum(pos.get('size', 0) * pos.get('mark_price', 0) for pos in positions)
        daily_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
        daily_pnl_percentage = (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Get risk metrics
        if positions:
            risk_request = {
                "positions": [
                    {
                        "symbol": pos["symbol"],
                        "quantity": pos["size"],
                        "entry_price": pos["entry_price"],
                        "current_price": pos["mark_price"],
                        "position_type": pos["side"].upper(),
                        "entry_timestamp": datetime.now().isoformat()
                    }
                    for pos in positions
                ],
                "portfolio_value": portfolio_value
            }
            
            risk_response = await risk_client.post("/calculate_portfolio_risk", json=risk_request)
            risk_metrics = risk_response.json() if risk_response.status_code == 200 else {}
        else:
            risk_metrics = {}
        
        # Top performers
        top_performers = sorted(
            [
                {
                    "symbol": pos["symbol"],
                    "pnl": pos.get("unrealized_pnl", 0),
                    "percentage": pos.get("percentage", 0)
                }
                for pos in positions
            ],
            key=lambda x: x["pnl"],
            reverse=True
        )[:5]
        
        # Recent trades
        recent_trades = [
            {
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "price": order.get("average_price") or order.get("price"),
                "timestamp": order["timestamp"],
                "status": order["status"]
            }
            for order in orders[-10:]
        ]
        
        return DashboardData(
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_percentage,
            total_positions=len(positions),
            active_orders=len([o for o in orders if o["status"] in ["SUBMITTED", "PENDING"]]),
            risk_metrics=risk_metrics,
            top_performers=top_performers,
            recent_trades=recent_trades
        )
        
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market data endpoints
@app.get("/api/market/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    """Get market data for a symbol."""
    try:
        # Get from cache first
        cached_data = redis_client.get(f"market:{symbol}")
        if cached_data:
            data = json.loads(cached_data)
            return MarketDataResponse(**data)
        
        # Fetch from database
        market_data = await db_manager.get_latest_market_data(symbol)
        if not market_data:
            raise HTTPException(status_code=404, detail=f"Market data not found for {symbol}")
        
        response = MarketDataResponse(
            symbol=symbol,
            price=market_data["close"],
            change=market_data["change"],
            change_percentage=market_data["change_percentage"],
            volume=market_data["volume"],
            timestamp=market_data["timestamp"]
        )
        
        # Cache for 5 seconds
        redis_client.setex(f"market:{symbol}", 5, response.model_dump_json())
        
        return response
        
    except Exception as e:
        logger.error(f"Market data error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Proxy endpoints to microservices
@app.post("/api/signals")
async def generate_signal(request: dict, current_user: User = Depends(get_current_user)):
    """Generate trading signal."""
    try:
        response = await decision_client.post("/generate_signal", json=request)
        return response.json()
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk/calculate")
async def calculate_risk(request: dict, current_user: User = Depends(get_current_user)):
    """Calculate portfolio risk."""
    try:
        response = await risk_client.post("/calculate_portfolio_risk", json=request)
        return response.json()
    except Exception as e:
        logger.error(f"Risk calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/orders")
async def create_order(request: dict, current_user: User = Depends(get_current_user)):
    """Create trading order."""
    try:
        response = await execution_client.post("/orders", json=request)
        return response.json()
    except Exception as e:
        logger.error(f"Order creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orders")
async def get_orders(current_user: User = Depends(get_current_user)):
    """Get user orders."""
    try:
        response = await execution_client.get("/orders")
        return response.json()
    except Exception as e:
        logger.error(f"Get orders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions(current_user: User = Depends(get_current_user)):
    """Get user positions."""
    try:
        response = await execution_client.get("/positions")
        return response.json()
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time updates
@app.websocket("/ws/market")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.ping()
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("Market data WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Market data WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Background tasks
async def broadcast_market_data():
    """Broadcast market data to WebSocket clients."""
    while True:
        try:
            if websocket_connections:
                # Get latest market data
                symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
                market_data = {}
                
                for symbol in symbols:
                    try:
                        data = await db_manager.get_latest_market_data(symbol)
                        if data:
                            market_data[symbol] = {
                                "symbol": symbol,
                                "price": data["close"],
                                "change": data["change"],
                                "change_percentage": data["change_percentage"],
                                "volume": data["volume"],
                                "timestamp": data["timestamp"].isoformat()
                            }
                    except Exception as e:
                        logger.error(f"Error fetching market data for {symbol}: {e}")
                
                # Broadcast to all clients
                if market_data:
                    message = {
                        "type": "market_data",
                        "data": market_data
                    }
                    
                    disconnected = []
                    for websocket in websocket_connections:
                        try:
                            await websocket.send_text(json.dumps(message))
                        except Exception:
                            disconnected.append(websocket)
                    
                    # Remove disconnected clients
                    for websocket in disconnected:
                        websocket_connections.remove(websocket)
            
            await asyncio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Error broadcasting market data: {e}")
            await asyncio.sleep(5)

# Serve static files (React app)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Catch-all route for React app
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_react_app(full_path: str):
    """Serve React application."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QuantTrade Pro</title>
        <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
    </head>
    <body>
        <div id="root"></div>
        <script src="/static/js/main.js"></script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
