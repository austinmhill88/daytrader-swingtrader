"""
Configuration management with environment variable resolution.
"""
import yaml
import os
import re
from typing import Any, Dict
from pathlib import Path


def resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variables in strings like ${VAR_NAME}.
    
    Args:
        value: Value to process (can be str, dict, list, or other)
        
    Returns:
        Processed value with environment variables resolved
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    
    return value


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and resolve environment variables.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve all environment variables in the config
    config = resolve_env_vars(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration fields are present and valid.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        'alpaca.key_id',
        'alpaca.secret_key',
        'alpaca.base_url',
    ]
    
    for field in required_fields:
        keys = field.split('.')
        value = config
        try:
            for key in keys:
                value = value[key]
            
            if not value or value == "":
                raise ValueError(f"Required configuration field '{field}' is empty")
                
        except KeyError:
            raise ValueError(f"Required configuration field '{field}' is missing")
    
    # Validate numeric ranges
    if config.get('risk', {}).get('daily_max_drawdown_pct', 0) <= 0:
        raise ValueError("daily_max_drawdown_pct must be > 0")
    
    if config.get('risk', {}).get('per_trade_risk_pct', 0) <= 0:
        raise ValueError("per_trade_risk_pct must be > 0")


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation path.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'risk.daily_max_drawdown_pct')
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    keys = path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
