"""
AI Client - Interface for trading system to communicate with AI server.
"""
import json
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None


class AIClient:
    """
    Client for communicating with local AI server.
    Provides analysis, suggestions, and reasoning for trades.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AI client.
        
        Args:
            config: Configuration dictionary with ai_server settings
        """
        if httpx is None:
            logger.warning("httpx not installed, AI features disabled")
            self.enabled = False
            return
        
        ai_config = config.get('ai_server', {})
        self.enabled = ai_config.get('enabled', False)
        self.base_url = ai_config.get('url', 'http://127.0.0.1:8000')
        self.model = ai_config.get('model', 'qwen2.5:3b-instruct')
        self.timeout = ai_config.get('timeout', 30)
        
        if self.enabled:
            logger.info(f"AI Client initialized - Server: {self.base_url}, Model: {self.model}")
        else:
            logger.info("AI Client disabled by configuration")
    
    async def _check_health(self) -> bool:
        """Check if AI server is healthy."""
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"AI server health check failed: {e}")
            return False
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send chat messages to AI server.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response (not implemented yet)
            options: Optional runtime options
            
        Returns:
            AI response text
        """
        if not self.enabled:
            return ""
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "options": options or {},
                        "stream": False  # Non-streaming for now
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("message", {}).get("content", "")
                else:
                    logger.error(f"AI chat failed: {response.status_code} - {response.text}")
                    return ""
        except Exception as e:
            logger.error(f"Error calling AI chat: {e}")
            return ""
    
    async def analyze_trade_signal(
        self, 
        signal: Dict[str, Any], 
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a trade signal and provide reasoning.
        
        Args:
            signal: Trade signal information
            market_context: Optional market context
            
        Returns:
            Analysis dict with reasoning, confidence, and suggestions
        """
        if not self.enabled:
            return None
        
        # Build prompt
        prompt_parts = [
            f"Analyze this trading signal:",
            f"Symbol: {signal.get('symbol')}",
            f"Strategy: {signal.get('strategy_name')}",
            f"Direction: {signal.get('direction')}",
            f"Strength: {signal.get('strength')}",
            f"Confidence: {signal.get('confidence')}",
            f"Reason: {signal.get('reason')}",
        ]
        
        if market_context:
            prompt_parts.append("\nMarket Context:")
            for key, value in market_context.items():
                prompt_parts.append(f"{key}: {value}")
        
        prompt_parts.append(
            "\nProvide brief analysis (2-3 sentences): "
            "1) Key risks, 2) Confidence assessment, 3) Timing considerations"
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are a trading analysis assistant. Provide concise, "
                          "actionable insights for trade signals."
            },
            {
                "role": "user",
                "content": "\n".join(prompt_parts)
            }
        ]
        
        try:
            response = await self.chat(
                messages, 
                options={"num_predict": 150, "temperature": 0.2}
            )
            
            if response:
                return {
                    "analysis": response,
                    "model": self.model,
                    "timestamp": None
                }
        except Exception as e:
            logger.error(f"Error analyzing trade signal: {e}")
        
        return None
    
    async def summarize_daily_performance(
        self, 
        performance: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate daily performance summary.
        
        Args:
            performance: Performance metrics
            
        Returns:
            Summary text
        """
        if not self.enabled:
            return None
        
        prompt = f"""Summarize today's trading performance (2-3 sentences):

Performance Metrics:
- P&L: ${performance.get('pnl', 0):.2f} ({performance.get('pnl_pct', 0):.2f}%)
- Trades: {performance.get('num_trades', 0)}
- Win Rate: {performance.get('win_rate', 0):.1f}%
- Largest Win: ${performance.get('largest_win', 0):.2f}
- Largest Loss: ${performance.get('largest_loss', 0):.2f}

Provide: 1) Overall assessment, 2) Key strength/weakness, 3) Tomorrow's focus"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a trading performance analyst. "
                          "Provide concise, objective summaries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            return await self.chat(
                messages, 
                options={"num_predict": 120, "temperature": 0.3}
            )
        except Exception as e:
            logger.error(f"Error summarizing performance: {e}")
            return None
    
    async def analyze_risk_event(
        self, 
        event_type: str, 
        details: Dict[str, Any]
    ) -> Optional[str]:
        """
        Analyze a risk event.
        
        Args:
            event_type: Type of risk event (kill_switch, drawdown, etc.)
            details: Event details
            
        Returns:
            Analysis and recommendations
        """
        if not self.enabled:
            return None
        
        prompt = f"""Risk Event Alert:

Event Type: {event_type}
Details:
"""
        for key, value in details.items():
            prompt += f"- {key}: {value}\n"
        
        prompt += "\nProvide: 1) Root cause assessment, 2) Immediate actions, 3) Prevention strategy"
        
        messages = [
            {
                "role": "system",
                "content": "You are a risk management analyst. "
                          "Provide clear, actionable guidance for risk events."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            return await self.chat(
                messages, 
                options={"num_predict": 180, "temperature": 0.2}
            )
        except Exception as e:
            logger.error(f"Error analyzing risk event: {e}")
            return None
