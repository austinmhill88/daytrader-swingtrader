from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class Bar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None

class OrderIntent(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str  # "market", "limit"
    limit_price: Optional[float] = None
    time_in_force: str = "day"
    bracket: Optional[Dict] = None  # {"take_profit": x, "stop_loss": y}

class Position(BaseModel):
    symbol: str
    qty: int
    avg_price: float
    market_price: float
    unrealized_pl: float

class Signal(BaseModel):
    symbol: str
    strength: float  # -1 to +1
    confidence: float  # 0 to 1
    reason: str