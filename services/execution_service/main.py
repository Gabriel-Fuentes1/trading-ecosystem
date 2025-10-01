"""
Execution Service - Order Management and Trade Execution
Microservice for handling trade execution and order management.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import logging
import json
from datetime import datetime, timedelta
from enum import Enum
import uuid
from contextlib import asynccontextmanager

# Trading libraries
import ccxt
import websockets

# Custom imports
from models.order_models import Order, OrderStatus, OrderType
from utils.exchange_manager import ExchangeManager
from utils.database import DatabaseManager
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()

# Global variables
exchange_manager = None
db_manager = None
active_orders = {}
websocket_connections = []

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderTypeEnum(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global exchange_manager, db_manager
    
    try:
        # Initialize components
        exchange_manager = ExchangeManager(settings.EXCHANGE_CONFIG)
        db_manager = DatabaseManager(settings.DATABASE_URL)
        
        # Start background tasks
        asyncio.create_task(monitor_orders())
        asyncio.create_task(process_order_queue())
        
        logger.info("Execution service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize execution service: {e}")
        raise
    finally:
        # Cleanup
        if db_manager:
            await db_manager.close()
        logger.info("Execution service shutting down")

app = FastAPI(
    title="Trading Execution Service",
    description="Microservice for trade execution and order management",
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
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price (required for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (for stop orders)")
    time_in_force: str = Field(default="GTC", description="Time in force (GTC, IOC, FOK)")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in ['limit', 'stop_loss', 'take_profit'] and v is None:
            raise ValueError('Price is required for limit and stop orders')
        return v

class OrderResponse(BaseModel):
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    commission: float
    timestamp: datetime
    exchange_order_id: Optional[str]

class PositionInfo(BaseModel):
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    margin: float
    timestamp: datetime

class ExecutionReport(BaseModel):
    order_id: str
    execution_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    trade_id: str

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check exchange connection
        exchange_status = await exchange_manager.check_connection()
        
        # Check database connection
        await db_manager.test_connection()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "service": "execution_service",
            "version": "1.0.0",
            "exchange_status": exchange_status,
            "active_orders": len(active_orders)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/orders", response_model=OrderResponse)
async def create_order(order_request: OrderRequest):
    """Create a new trading order."""
    try:
        logger.info(f"Creating {order_request.side} order for {order_request.symbol}")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        client_order_id = order_request.client_order_id or order_id
        
        # Validate order parameters
        await validate_order(order_request)
        
        # Create order object
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        # Store order in database
        await db_manager.save_order(order)
        
        # Add to active orders
        active_orders[order_id] = order
        
        # Submit order to exchange
        try:
            exchange_result = await exchange_manager.create_order(
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force,
                client_order_id=client_order_id
            )
            
            # Update order with exchange information
            order.exchange_order_id = exchange_result.get('id')
            order.status = OrderStatus.SUBMITTED
            
            await db_manager.update_order(order)
            
        except Exception as e:
            logger.error(f"Failed to submit order to exchange: {e}")
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            await db_manager.update_order(order)
            
            # Remove from active orders
            active_orders.pop(order_id, None)
            
            raise HTTPException(status_code=400, detail=f"Order rejected: {str(e)}")
        
        # Create response
        response = OrderResponse(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.quantity - order.filled_quantity,
            average_price=order.average_price,
            commission=order.commission,
            timestamp=order.timestamp,
            exchange_order_id=order.exchange_order_id
        )
        
        # Notify WebSocket clients
        await notify_order_update(response)
        
        logger.info(f"Order {order_id} created successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get order details by ID."""
    try:
        order = await db_manager.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return OrderResponse(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.quantity - order.filled_quantity,
            average_price=order.average_price,
            commission=order.commission,
            timestamp=order.timestamp,
            exchange_order_id=order.exchange_order_id
        )
        
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an existing order."""
    try:
        order = active_orders.get(order_id)
        if not order:
            order = await db_manager.get_order(order_id)
            if not order:
                raise HTTPException(status_code=404, detail="Order not found")
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel order with status {order.status}")
        
        # Cancel order on exchange
        if order.exchange_order_id:
            await exchange_manager.cancel_order(order.symbol, order.exchange_order_id)
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        
        await db_manager.update_order(order)
        
        # Remove from active orders
        active_orders.pop(order_id, None)
        
        # Notify WebSocket clients
        await notify_order_update(OrderResponse(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.quantity - order.filled_quantity,
            average_price=order.average_price,
            commission=order.commission,
            timestamp=order.timestamp,
            exchange_order_id=order.exchange_order_id
        ))
        
        logger.info(f"Order {order_id} cancelled successfully")
        return {"message": "Order cancelled successfully", "order_id": order_id}
        
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Field(default=100, le=1000)
):
    """Get list of orders with optional filters."""
    try:
        orders = await db_manager.get_orders(symbol=symbol, status=status, limit=limit)
        
        return [
            OrderResponse(
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                status=order.status,
                filled_quantity=order.filled_quantity,
                remaining_quantity=order.quantity - order.filled_quantity,
                average_price=order.average_price,
                commission=order.commission,
                timestamp=order.timestamp,
                exchange_order_id=order.exchange_order_id
            )
            for order in orders
        ]
        
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions", response_model=List[PositionInfo])
async def get_positions():
    """Get current positions."""
    try:
        positions = await exchange_manager.get_positions()
        
        return [
            PositionInfo(
                symbol=pos['symbol'],
                side=pos['side'],
                size=pos['size'],
                entry_price=pos['entryPrice'],
                mark_price=pos['markPrice'],
                unrealized_pnl=pos['unrealizedPnl'],
                percentage=pos['percentage'],
                margin=pos['initialMargin'],
                timestamp=datetime.now()
            )
            for pos in positions if pos['size'] != 0
        ]
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket):
    """WebSocket endpoint for real-time order updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.ping()
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Helper functions
async def validate_order(order_request: OrderRequest):
    """Validate order parameters."""
    # Check symbol validity
    if not await exchange_manager.is_valid_symbol(order_request.symbol):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {order_request.symbol}")
    
    # Check minimum quantity
    min_quantity = await exchange_manager.get_min_quantity(order_request.symbol)
    if order_request.quantity < min_quantity:
        raise HTTPException(status_code=400, detail=f"Quantity below minimum: {min_quantity}")
    
    # Check account balance
    balance = await exchange_manager.get_balance()
    # Add balance validation logic here

async def notify_order_update(order_response: OrderResponse):
    """Notify WebSocket clients of order updates."""
    if websocket_connections:
        message = {
            "type": "order_update",
            "data": order_response.model_dump()
        }
        
        # Send to all connected clients
        disconnected = []
        for websocket in websocket_connections:
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            websocket_connections.remove(websocket)

async def monitor_orders():
    """Background task to monitor order status."""
    while True:
        try:
            # Check active orders
            for order_id, order in list(active_orders.items()):
                if order.exchange_order_id:
                    # Get order status from exchange
                    exchange_order = await exchange_manager.get_order_status(
                        order.symbol, order.exchange_order_id
                    )
                    
                    # Update order if status changed
                    if exchange_order['status'] != order.status:
                        order.status = exchange_order['status']
                        order.filled_quantity = exchange_order.get('filled', 0)
                        order.average_price = exchange_order.get('average', order.price)
                        order.commission = exchange_order.get('fee', {}).get('cost', 0)
                        
                        await db_manager.update_order(order)
                        
                        # Remove from active orders if completed
                        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            active_orders.pop(order_id, None)
                        
                        # Notify clients
                        await notify_order_update(OrderResponse(
                            order_id=order.order_id,
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=order.quantity,
                            price=order.price,
                            status=order.status,
                            filled_quantity=order.filled_quantity,
                            remaining_quantity=order.quantity - order.filled_quantity,
                            average_price=order.average_price,
                            commission=order.commission,
                            timestamp=order.timestamp,
                            exchange_order_id=order.exchange_order_id
                        ))
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error monitoring orders: {e}")
            await asyncio.sleep(10)

async def process_order_queue():
    """Background task to process order queue."""
    while True:
        try:
            # Process any queued orders
            # This could be used for batch processing or retry logic
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing order queue: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
