"""
Logging utilities with structured logging support.
"""
from loguru import logger
import sys
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    logs_dir: str,
    level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "30 days",
    format_type: str = "json",
    enable_console: bool = True
) -> None:
    """
    Set up comprehensive logging with rotation and multiple outputs.
    
    Args:
        logs_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate logs (e.g., "1 day", "500 MB")
        retention: How long to keep logs (e.g., "30 days")
        format_type: Format type ("json" or "text")
        enable_console: Whether to log to console
    """
    # Create logs directory if it doesn't exist
    log_path = Path(logs_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Define format based on type
    if format_type == "json":
        log_format = (
            '{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler (with colors for text format)
    if enable_console:
        console_format = log_format if format_type == "json" else (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=console_format,
            level=level,
            colorize=(format_type != "json"),
            enqueue=True
        )
    
    # Runtime log (all messages at INFO and above)
    logger.add(
        os.path.join(logs_dir, "runtime.log"),
        format=log_format,
        level="INFO",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
        serialize=(format_type == "json")
    )
    
    # Error log (ERROR and above only)
    logger.add(
        os.path.join(logs_dir, "errors.log"),
        format=log_format,
        level="ERROR",
        rotation=rotation,
        retention="60 days",  # Keep errors longer
        compression="zip",
        enqueue=True,
        serialize=(format_type == "json")
    )
    
    # Debug log (only if level is DEBUG)
    if level == "DEBUG":
        logger.add(
            os.path.join(logs_dir, "debug.log"),
            format=log_format,
            level="DEBUG",
            rotation=rotation,
            retention="7 days",  # Debug logs for shorter period
            compression="zip",
            enqueue=True,
            serialize=(format_type == "json")
        )
    
    # Trading activity log (separate for trade-related events)
    logger.add(
        os.path.join(logs_dir, "trades.log"),
        format=log_format,
        level="INFO",
        rotation=rotation,
        retention="90 days",  # Keep trade logs longer for audit
        compression="zip",
        enqueue=True,
        serialize=(format_type == "json"),
        filter=lambda record: "TRADE" in record["message"] or "ORDER" in record["message"]
    )
    
    logger.info(f"Logging initialized: level={level}, dir={logs_dir}, format={format_type}")


def log_trade(action: str, symbol: str, qty: int, price: float, **kwargs) -> None:
    """
    Log a trade event with structured data.
    
    Args:
        action: Trade action (e.g., "BUY", "SELL", "FILLED")
        symbol: Stock symbol
        qty: Quantity
        price: Trade price
        **kwargs: Additional trade metadata
    """
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    msg = f"TRADE | {action} | {symbol} | qty={qty} | price={price:.2f}"
    if extra_info:
        msg += f" | {extra_info}"
    logger.info(msg)


def log_signal(strategy: str, symbol: str, strength: float, confidence: float, reason: str) -> None:
    """
    Log a trading signal.
    
    Args:
        strategy: Strategy name
        symbol: Stock symbol
        strength: Signal strength (-1 to +1)
        confidence: Signal confidence (0 to 1)
        reason: Signal reasoning
    """
    logger.info(
        f"SIGNAL | {strategy} | {symbol} | "
        f"strength={strength:.3f} | confidence={confidence:.3f} | {reason}"
    )


def log_error_with_context(error: Exception, context: str, **kwargs) -> None:
    """
    Log an error with additional context.
    
    Args:
        error: Exception object
        context: Context description
        **kwargs: Additional context metadata
    """
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    msg = f"ERROR | {context} | {type(error).__name__}: {str(error)}"
    if extra_info:
        msg += f" | {extra_info}"
    logger.error(msg)
    logger.exception(error)
