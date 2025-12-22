"""
Universe selection and analytics (Phase 1).
Dynamic universe building based on liquidity and quality metrics.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class UniverseAnalytics:
    """
    Universe selection with liquidity scoring and tier management.
    Phase 1 implementation - extensible for Phase 2-4 enhancements.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize universe analytics.
        
        Args:
            config: Universe configuration from config.yaml
        """
        self.config = config
        universe_config = config.get('universe', {})
        
        self.min_price = universe_config.get('min_price', 5.0)
        self.max_spread_bps = universe_config.get('max_spread_bps', 50)
        self.adv_lookback_days = universe_config.get('adv_lookback_days', 60)
        self.top_by_dollar_volume = universe_config.get('top_by_dollar_volume', 500)
        
        # Tier configuration
        tiers = universe_config.get('tiers', {})
        self.core_size = tiers.get('core', 200)
        self.extended_size = tiers.get('extended', 1000)
        
        # Analytics configuration
        analytics = universe_config.get('analytics', {})
        self.enable_liquidity_scoring = analytics.get('enable_liquidity_scoring', True)
        self.enable_volatility_bucketing = analytics.get('enable_volatility_bucketing', True)
        self.min_adv_usd = analytics.get('min_adv_usd', 1000000)
        
        logger.info(
            f"UniverseAnalytics initialized | "
            f"Core: {self.core_size}, Extended: {self.extended_size}"
        )
    
    def calculate_liquidity_score(
        self,
        symbol_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate liquidity scores for symbols.
        
        Args:
            symbol_data: Dictionary of symbol -> DataFrame with price/volume data
            
        Returns:
            DataFrame with liquidity metrics per symbol
        """
        try:
            results = []
            
            for symbol, df in symbol_data.items():
                if df is None or len(df) == 0:
                    continue
                
                # Calculate average dollar volume
                df['dollar_volume'] = df['close'] * df['volume']
                adv = df['dollar_volume'].mean()
                
                # Calculate spread (if available)
                if 'high' in df.columns and 'low' in df.columns:
                    df['spread'] = (df['high'] - df['low']) / df['close']
                    avg_spread_pct = df['spread'].mean() * 100
                    avg_spread_bps = avg_spread_pct * 100
                else:
                    avg_spread_bps = None
                
                # Calculate price stability
                price_std = df['close'].std()
                price_mean = df['close'].mean()
                price_cv = price_std / price_mean if price_mean > 0 else 0
                
                # Calculate volume consistency
                volume_std = df['volume'].std()
                volume_mean = df['volume'].mean()
                volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
                
                # Recent metrics
                recent_df = df.tail(20)  # Last 20 bars
                recent_adv = (recent_df['close'] * recent_df['volume']).mean()
                recent_price = recent_df['close'].mean()
                
                # Liquidity score (0-100)
                # Higher is better
                score = 0.0
                
                # ADV component (40 points)
                if adv >= self.min_adv_usd:
                    adv_score = min(40, (adv / self.min_adv_usd) * 10)
                    score += adv_score
                
                # Spread component (30 points)
                if avg_spread_bps is not None:
                    if avg_spread_bps <= self.max_spread_bps:
                        spread_score = 30 * (1 - avg_spread_bps / self.max_spread_bps)
                        score += spread_score
                
                # Volume consistency (15 points)
                if volume_cv < 2.0:  # Less than 200% coefficient of variation
                    consistency_score = 15 * (1 - min(volume_cv / 2.0, 1.0))
                    score += consistency_score
                
                # Price stability (15 points)
                if price_cv < 0.5:
                    stability_score = 15 * (1 - min(price_cv / 0.5, 1.0))
                    score += stability_score
                
                results.append({
                    'symbol': symbol,
                    'adv_usd': adv,
                    'recent_adv_usd': recent_adv,
                    'avg_spread_bps': avg_spread_bps,
                    'avg_price': price_mean,
                    'recent_price': recent_price,
                    'price_cv': price_cv,
                    'volume_cv': volume_cv,
                    'liquidity_score': score,
                    'num_bars': len(df)
                })
            
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('liquidity_score', ascending=False)
            
            logger.info(f"Calculated liquidity scores for {len(df_results)} symbols")
            return df_results
            
        except Exception as e:
            logger.error(f"Error calculating liquidity scores: {e}")
            return pd.DataFrame()
    
    def build_universe(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        tier: str = 'core'
    ) -> List[str]:
        """
        Build trading universe based on liquidity and quality.
        
        Args:
            symbol_data: Dictionary of symbol -> DataFrame
            tier: 'core' or 'extended'
            
        Returns:
            List of selected symbols
        """
        try:
            # Calculate liquidity scores
            liquidity_df = self.calculate_liquidity_score(symbol_data)
            
            if len(liquidity_df) == 0:
                logger.warning("No symbols passed liquidity analysis")
                return []
            
            # Filter by minimum criteria
            filtered = liquidity_df[
                (liquidity_df['adv_usd'] >= self.min_adv_usd) &
                (liquidity_df['avg_price'] >= self.min_price)
            ]
            
            if 'avg_spread_bps' in filtered.columns:
                filtered = filtered[
                    (filtered['avg_spread_bps'].isna()) |
                    (filtered['avg_spread_bps'] <= self.max_spread_bps)
                ]
            
            # Select top N by liquidity score
            tier_size = self.core_size if tier == 'core' else self.extended_size
            selected = filtered.head(tier_size)
            
            symbols = selected['symbol'].tolist()
            
            logger.info(
                f"Built {tier} universe | "
                f"{len(symbols)} symbols selected from {len(liquidity_df)} analyzed"
            )
            
            if len(selected) > 0:
                logger.info(
                    f"Universe stats | "
                    f"Avg ADV: ${selected['adv_usd'].mean()/1e6:.1f}M, "
                    f"Avg Score: {selected['liquidity_score'].mean():.1f}"
                )
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error building universe: {e}")
            return []
    
    def bucket_by_volatility(
        self,
        symbol_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[str]]:
        """
        Bucket symbols by volatility regime.
        
        Args:
            symbol_data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dictionary of bucket -> list of symbols
        """
        try:
            results = []
            
            for symbol, df in symbol_data.items():
                if df is None or len(df) == 0:
                    continue
                
                # Calculate realized volatility
                df['returns'] = df['close'].pct_change()
                realized_vol = df['returns'].std() * np.sqrt(252)  # Annualized
                
                results.append({
                    'symbol': symbol,
                    'realized_vol': realized_vol
                })
            
            df_results = pd.DataFrame(results)
            
            # Define buckets
            df_results['bucket'] = pd.cut(
                df_results['realized_vol'],
                bins=[0, 0.20, 0.35, 1.0],
                labels=['low_vol', 'medium_vol', 'high_vol']
            )
            
            # Group by bucket
            buckets = {}
            for bucket_name in ['low_vol', 'medium_vol', 'high_vol']:
                bucket_symbols = df_results[df_results['bucket'] == bucket_name]['symbol'].tolist()
                buckets[bucket_name] = bucket_symbols
                logger.info(f"Volatility bucket '{bucket_name}': {len(bucket_symbols)} symbols")
            
            return buckets
            
        except Exception as e:
            logger.error(f"Error bucketing by volatility: {e}")
            return {}
    
    def get_universe_summary(self, symbols: List[str]) -> Dict:
        """
        Get summary statistics for a universe.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary with summary stats
        """
        return {
            'num_symbols': len(symbols),
            'symbols': symbols[:10] + (['...'] if len(symbols) > 10 else [])
        }
    
    def filter_earnings_blackout(
        self,
        symbols: List[str],
        alpaca_client=None,
        blackout_days_before: int = 2,
        blackout_days_after: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Filter out symbols in earnings blackout period.
        
        Args:
            symbols: List of symbols to filter
            alpaca_client: AlpacaClient instance for earnings calendar
            blackout_days_before: Days before earnings to blackout
            blackout_days_after: Days after earnings to blackout
            
        Returns:
            Tuple of (tradeable_symbols, blackout_symbols)
        """
        if alpaca_client is None:
            # Can't check without client, return all as tradeable
            logger.warning("No Alpaca client provided for earnings blackout check")
            return symbols, []
        
        
        tradeable = []
        blackout = []
        
        today = datetime.now().date()
        blackout_start = today - timedelta(days=blackout_days_before)
        blackout_end = today + timedelta(days=blackout_days_after)
        
        for symbol in symbols:
            try:
                # Check if symbol has earnings within blackout window
                # Using total blackout window (before + after)
                total_days = blackout_days_before + blackout_days_after
                has_earnings = alpaca_client.has_upcoming_earnings(symbol, days_ahead=total_days)
                
                if has_earnings:
                    blackout.append(symbol)
                    logger.debug(f"Blackout: {symbol} has earnings within {total_days} days")
                else:
                    tradeable.append(symbol)
                
            except Exception as e:
                logger.warning(f"Error checking earnings for {symbol}: {e}")
                # Conservative: add to tradeable if can't check
                tradeable.append(symbol)
        
        if blackout:
            logger.info(
                f"Earnings blackout | "
                f"{len(blackout)} symbols in blackout period: {', '.join(blackout[:5])}"
            )
        
        return tradeable, blackout
    
    def filter_shortable(
        self,
        symbols: List[str],
        alpaca_client=None
    ) -> Tuple[List[str], List[str]]:
        """
        Filter symbols by shortability.
        
        Args:
            symbols: List of symbols to filter
            alpaca_client: AlpacaClient instance for shortability checks
            
        Returns:
            Tuple of (shortable_symbols, non_shortable_symbols)
        """
        if alpaca_client is None:
            logger.warning("No Alpaca client provided for shortability check")
            return symbols, []
        
        shortable = []
        non_shortable = []
        
        for symbol in symbols:
            try:
                if alpaca_client.is_shortable(symbol):
                    shortable.append(symbol)
                else:
                    non_shortable.append(symbol)
            except Exception as e:
                logger.warning(f"Error checking shortability for {symbol}: {e}")
                # Conservative: add to shortable if can't check
                shortable.append(symbol)
        
        logger.info(
            f"Shortability check | "
            f"{len(shortable)}/{len(symbols)} symbols are shortable "
            f"({100*len(shortable)/len(symbols):.1f}%)"
        )
        
        if non_shortable:
            logger.debug(
                f"Non-shortable symbols: {', '.join(non_shortable[:10])}"
                + ("..." if len(non_shortable) > 10 else "")
            )
        
        return shortable, non_shortable
    
    def build_universe_with_filters(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        tier: str = 'core',
        alpaca_client=None,
        apply_earnings_filter: bool = True,
        apply_shortability_filter: bool = False,
        previous_universe: Optional[List[str]] = None
    ) -> Dict:
        """
        Build universe with earnings and shortability filters.
        
        Args:
            symbol_data: Dictionary of symbol -> DataFrame
            tier: 'core' or 'extended'
            alpaca_client: AlpacaClient for filtering
            apply_earnings_filter: Filter earnings blackout periods
            apply_shortability_filter: Filter non-shortable symbols
            previous_universe: Previous universe for change tracking
            
        Returns:
            Dictionary with universe and metadata
        """
        # Build base universe
        base_universe = self.build_universe(symbol_data, tier)
        
        # Apply earnings filter
        if apply_earnings_filter and alpaca_client:
            tradeable, blackout = self.filter_earnings_blackout(
                base_universe, alpaca_client
            )
        else:
            tradeable = base_universe
            blackout = []
        
        # Apply shortability filter
        if apply_shortability_filter and alpaca_client:
            shortable, non_shortable = self.filter_shortable(
                tradeable, alpaca_client
            )
            final_universe = shortable
        else:
            final_universe = tradeable
            shortable = tradeable
            non_shortable = []
        
        # Track membership changes
        if previous_universe is not None:
            added = set(final_universe) - set(previous_universe)
            removed = set(previous_universe) - set(final_universe)
            
            if added:
                logger.info(
                    f"Universe additions ({len(added)}): {', '.join(list(added)[:5])}"
                    + ("..." if len(added) > 5 else "")
                )
            
            if removed:
                logger.info(
                    f"Universe removals ({len(removed)}): {', '.join(list(removed)[:5])}"
                    + ("..." if len(removed) > 5 else "")
                )
        
        return {
            'universe': final_universe,
            'base_count': len(base_universe),
            'earnings_blackout_count': len(blackout),
            'non_shortable_count': len(non_shortable),
            'final_count': len(final_universe),
            'blackout_symbols': blackout,
            'non_shortable_symbols': non_shortable,
            'tier': tier
        }
