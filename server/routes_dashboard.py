"""
Trading dashboard routes: summary KPIs, positions, and live prices.
Works against TradingController with defensive attribute access.
"""
from typing import Any, Dict, List
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import math

def _getattr_chain(obj: Any, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        try:
            cur = getattr(cur, part)
        except Exception:
            return default
    return cur if cur is not None else default

def _finite_or_none(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def _safe_num(x, default=0.0):
    """Return a finite float or default (use only for table rows, not KPIs)."""
    try:
        n = float(x)
        return n if math.isfinite(n) else default
    except Exception:
        return default

def _build_position_row(pos: Any, price_lookup: Dict[str, float]) -> Dict[str, Any]:
    # Try common fields: symbol, qty/quantity, avg_price, market_price
    symbol = getattr(pos, "symbol", None) or getattr(pos, "ticker", None) or getattr(pos, "name", "")
    qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None) or 0
    avg_price = getattr(pos, "avg_price", None) or getattr(pos, "average_price", None) or getattr(pos, "cost_basis", None) or 0.0
    mark = price_lookup.get(symbol, 0.0)

    qty_f = _safe_num(qty, 0.0)
    avg_f = _safe_num(avg_price, 0.0)
    mark_f = _safe_num(mark, 0.0)

    pnl_usd = (mark_f - avg_f) * qty_f if math.isfinite(mark_f) and math.isfinite(avg_f) and math.isfinite(qty_f) else 0.0
    pnl_pct = (mark_f / avg_f - 1.0) * 100.0 if avg_f > 0 else 0.0

    return {
        "symbol": symbol,
        "qty": qty_f,
        "avg_price": avg_f,
        "mark": mark_f,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
    }

def make_router(controller: Any):
    router = APIRouter()

    @router.get("/dashboard/summary")
    async def summary():
        try:
            bot = _getattr_chain(controller, "_bot", None)

            # Risk caps from config (best-effort)
            cfg_path = getattr(controller, "config_path", "config/config.yaml")
            risk_caps = {"max_dd": "2%", "per_trade": "0.4%", "kill_switch": True, "exposure_limits": "default"}
            try:
                import yaml, pathlib
                p = pathlib.Path(cfg_path)
                if p.exists():
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    r = data.get("risk", {}) or {}
                    risk_caps = {
                        "max_dd": r.get("max_drawdown", "2%"),
                        "per_trade": r.get("per_trade", "0.4%"),
                        "kill_switch": bool(r.get("kill_switch", True)),
                        "exposure_limits": r.get("exposure_limits", "default"),
                    }
            except Exception:
                pass

            # Prefer broker-reported account values when available, even if market is closed
            acct = _getattr_chain(controller, "alpaca", None) or _getattr_chain(bot, "alpaca", None)
            broker_equity = broker_cash = broker_bp = None
            if acct and hasattr(acct, "get_account"):
                try:
                    a = acct.get_account()
                    broker_equity = _finite_or_none(getattr(a, "equity", None))
                    broker_cash = _finite_or_none(getattr(a, "cash", None))
                    broker_bp = _finite_or_none(getattr(a, "buying_power", None))
                except Exception:
                    pass

            # Portfolio fallbacks
            portfolio_equity = _finite_or_none(_getattr_chain(bot, "portfolio.equity", None))
            portfolio_cash = _finite_or_none(_getattr_chain(bot, "portfolio.cash", None))
            portfolio_bp = _finite_or_none(_getattr_chain(bot, "portfolio.buying_power", None))

            realized_pnl = _finite_or_none(_getattr_chain(bot, "portfolio.realized_pnl", None))
            unrealized_pnl = _finite_or_none(_getattr_chain(bot, "portfolio.unrealized_pnl", None))
            drawdown_pct = _finite_or_none(_getattr_chain(bot, "risk_manager.current_drawdown_pct", None)) or _finite_or_none(_getattr_chain(bot, "risk_manager.current_drawdown", None))
            exposure_pct = _finite_or_none(_getattr_chain(bot, "risk_manager.current_exposure_pct", None)) or _finite_or_none(_getattr_chain(bot, "risk_manager.exposure_pct", None))
            regime = _getattr_chain(bot, "regime_detector.current_regime", None) or "unknown"

            # Counts
            positions = _getattr_chain(bot, "portfolio.positions", None)
            positions_count = len(positions) if isinstance(positions, (list, tuple)) else (len(positions) if isinstance(positions, dict) else 0)
            open_orders = _getattr_chain(bot, "execution_engine.open_orders", None)
            open_orders_count = len(open_orders) if isinstance(open_orders, (list, tuple)) else (len(open_orders) if isinstance(open_orders, dict) else 0)

            # Assemble with sanitization: use broker values first, fallback to portfolio, else None
            d = {
                "equity": broker_equity if broker_equity is not None else portfolio_equity,
                "cash": broker_cash if broker_cash is not None else portfolio_cash,
                "buying_power": broker_bp if broker_bp is not None else portfolio_bp,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "drawdown_pct": drawdown_pct,
                "exposure_pct": exposure_pct,
                "regime": regime,
                "positions_count": positions_count,
                "open_orders_count": open_orders_count,
                "risk_caps": risk_caps,
            }
            # Final cleanup: any lingering non-finites → None
            for k, v in list(d.items()):
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    d[k] = None

            return d
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/dashboard/positions")
    async def positions():
        try:
            bot = _getattr_chain(controller, "_bot", None)
            raw_positions = _getattr_chain(bot, "portfolio.positions", []) or []
            # Figure out symbols
            symbols: List[str] = []
            for p in raw_positions:
                sym = getattr(p, "symbol", None) or getattr(p, "ticker", None)
                if sym:
                    symbols.append(sym)
            # Price lookup from data feed or alpaca
            prices: Dict[str, float] = {}
            data_feed = _getattr_chain(bot, "data_feed", None)
            alpaca = _getattr_chain(bot, "alpaca", None)
            for s in symbols:
                price = None
                if data_feed and hasattr(data_feed, "get_last_price"):
                    try:
                        price = data_feed.get_last_price(s)
                    except Exception:
                        price = None
                if price is None and alpaca:
                    try:
                        # Try last trade price via Alpaca if available
                        t = alpaca.get_last_trade(s)
                        price = float(getattr(t, "price", None) or getattr(t, "Price", None) or 0.0)
                    except Exception:
                        price = None
                prices[s] = float(price or 0.0)

            rows = [_build_position_row(p, prices) for p in raw_positions]
            return {"positions": rows}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/dashboard/tickers")
    async def tickers(symbols: str = Query("AAPL,MSFT,AMZN,NVDA,GOOGL,TSLA,META,SPY,QQQ")):
        try:
            bot = _getattr_chain(controller, "_bot", None)
            data_feed = _getattr_chain(bot, "data_feed", None)
            alpaca = _getattr_chain(bot, "alpaca", None)
            out: List[Dict[str, Any]] = []
            for s in [x.strip().upper() for x in symbols.split(",") if x.strip()]:
                last = None
                prev_close = None
                vol = None
                # Try data_feed if available
                if data_feed:
                    try:
                        last = data_feed.get_last_price(s)
                    except Exception:
                        last = None
                    try:
                        bars = data_feed.get_bars(s, "1Day", limit=2)
                        if isinstance(bars, list) and len(bars) >= 2:
                            prev_close = float(getattr(bars[-2], "close", None) or getattr(bars[-2], "c", None) or 0.0)
                    except Exception:
                        prev_close = None
                # Fallback to Alpaca
                if (last is None or prev_close is None) and alpaca:
                    try:
                        lt = alpaca.get_last_trade(s)
                        last = float(getattr(lt, "price", None) or 0.0)
                    except Exception:
                        pass
                    try:
                        bd = alpaca.get_latest_bar(s)
                        prev_close = float(getattr(bd, "close", None) or 0.0)
                        vol = int(getattr(bd, "volume", None) or 0)
                    except Exception:
                        pass

                last = float(last or 0.0)
                prev_close = float(prev_close or last or 0.0)
                change = last - prev_close
                change_pct = ((last / prev_close) - 1.0) * 100.0 if prev_close > 0 else 0.0
                out.append({
                    "symbol": s,
                    "last": round(last, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": vol,
                })
            return {"tickers": out}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router