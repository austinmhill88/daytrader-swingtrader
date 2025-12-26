"""
Data storage layer with Parquet support (Phase 1).
Handles partitioned storage by symbol and date with versioning.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import pandas as pd
from loguru import logger
import hashlib
import json


class DataStorage:
    """
    Data storage manager with Parquet support and versioning.
    Phase 1 implementation - extensible for Phase 2 feature store.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data storage.
        
        Args:
            config: Storage configuration from config.yaml
        """
        self.config = config
        storage_config = config.get('data_storage', {})
        
        self.format = storage_config.get('format', 'parquet')
        self.base_path = Path(storage_config.get('base_path', './data'))
        self.parquet_dir = Path(config.get('storage', {}).get('parquet_dir', './data/parquet'))
        self.compression = storage_config.get('compression', 'snappy')
        self.partition_by_symbol = storage_config.get('partitioning', {}).get('by_symbol', True)
        self.partition_by_date = storage_config.get('partitioning', {}).get('by_date', True)
        self.enable_versioning = storage_config.get('enable_versioning', True)
        
        # Create directories
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        (self.base_path / 'metadata').mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"DataStorage initialized | "
            f"Format: {self.format}, Path: {self.parquet_dir}, "
            f"Compression: {self.compression}"
        )
    
    def save_bars(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save bar data to partitioned Parquet.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            timeframe: Bar timeframe
            metadata: Optional metadata dict
            
        Returns:
            True if successful
        """
        try:
            if df is None or len(df) == 0:
                logger.warning(f"Empty dataframe for {symbol}, skipping save")
                return False
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            
            # Add symbol if not present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # Extract date for partitioning
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Build path based on partitioning strategy
            if self.partition_by_symbol and self.partition_by_date:
                # symbol=XXX/date=YYYY-MM-DD/
                path_parts = []
                for _, group in df.groupby(['symbol', 'date']):
                    sym = group['symbol'].iloc[0]
                    dt = group['date'].iloc[0]
                    partition_path = self.parquet_dir / f"symbol={sym}" / f"date={dt}"
                    partition_path.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    filepath = partition_path / filename
                    
                    # Save partition
                    group_df = group.drop(columns=['date'])
                    group_df.to_parquet(
                        filepath,
                        compression=self.compression,
                        index=False
                    )
                    path_parts.append(str(filepath))
                
                logger.info(f"Saved {len(df)} bars for {symbol} across {len(path_parts)} partitions")
                
            elif self.partition_by_symbol:
                # symbol=XXX/
                partition_path = self.parquet_dir / f"symbol={symbol}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                filepath = partition_path / filename
                
                df_save = df.drop(columns=['date']) if 'date' in df.columns else df
                df_save.to_parquet(filepath, compression=self.compression, index=False)
                logger.info(f"Saved {len(df)} bars for {symbol} to {filepath}")
            
            else:
                # No partitioning
                filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                filepath = self.parquet_dir / filename
                
                df_save = df.drop(columns=['date']) if 'date' in df.columns else df
                df_save.to_parquet(filepath, compression=self.compression, index=False)
                logger.info(f"Saved {len(df)} bars for {symbol} to {filepath}")
            
            # Save metadata if versioning enabled
            if self.enable_versioning and metadata:
                self._save_metadata(symbol, timeframe, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving bars for {symbol}: {e}")
            return False
    
    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load bar data from Parquet.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            dfs = []
            
            if self.partition_by_symbol:
                symbol_dir = self.parquet_dir / f"symbol={symbol}"
                if not symbol_dir.exists():
                    logger.warning(f"No data found for {symbol}")
                    return None
                
                # Find all parquet files for this symbol
                parquet_files = list(symbol_dir.rglob(f"{timeframe}_*.parquet"))
                
                if not parquet_files:
                    logger.warning(f"No {timeframe} data found for {symbol}")
                    return None
                
                for file in parquet_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
            
            else:
                # No partitioning - search by pattern
                pattern = f"{symbol}_{timeframe}_*.parquet"
                parquet_files = list(self.parquet_dir.glob(pattern))
                
                for file in parquet_files:
                    df = pd.read_parquet(file)
                    dfs.append(df)
            
            if not dfs:
                return None
            
            # Combine all dataframes
            combined = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp
            if 'timestamp' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                combined = combined.sort_values('timestamp')
                
                # Filter by date range if provided
                if start_date:
                    combined = combined[combined['timestamp'].dt.date >= start_date]
                if end_date:
                    combined = combined[combined['timestamp'].dt.date <= end_date]
            
            # Remove duplicates
            combined = combined.drop_duplicates(subset=['timestamp', 'symbol'])
            
            logger.info(f"Loaded {len(combined)} bars for {symbol}")
            return combined
            
        except Exception as e:
            logger.error(f"Error loading bars for {symbol}: {e}")
            return None
    
    def _save_metadata(self, symbol: str, timeframe: str, metadata: Dict) -> None:
        """Save metadata with versioning."""
        try:
            metadata_file = self.base_path / 'metadata' / f"{symbol}_{timeframe}_metadata.json"
            
            # Add version info
            metadata['version'] = datetime.now().isoformat()
            metadata['hash'] = hashlib.md5(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def list_available_symbols(self) -> List[str]:
        """
        Get list of symbols with stored data.
        
        Returns:
            List of symbol strings
        """
        try:
            symbols = set()
            
            if self.partition_by_symbol:
                # Look for symbol=XXX directories
                for path in self.parquet_dir.iterdir():
                    if path.is_dir() and path.name.startswith('symbol='):
                        symbol = path.name.split('=')[1]
                        symbols.add(symbol)
            else:
                # Parse filenames
                for file in self.parquet_dir.glob('*.parquet'):
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        symbols.add(parts[0])
            
            return sorted(list(symbols))
            
        except Exception as e:
            logger.error(f"Error listing symbols: {e}")
            return []
    
    def get_date_range(self, symbol: str, timeframe: str) -> Optional[tuple]:
        """
        Get date range for symbol's stored data.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            
        Returns:
            Tuple of (start_date, end_date) or None
        """
        try:
            df = self.load_bars(symbol, timeframe)
            if df is not None and len(df) > 0 and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return (
                    df['timestamp'].min().date(),
                    df['timestamp'].max().date()
                )
            return None
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None
