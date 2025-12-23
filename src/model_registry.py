"""
Simple local model registry: track promoted model artifacts by strategy.
Stores registry.json in storage/model_dir.
"""
import json
from pathlib import Path
from typing import Optional, Dict
from loguru import logger


class ModelRegistry:
    """
    Local model registry for tracking promoted models.
    Stores metadata in registry.json file.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize model registry.
        
        Args:
            model_dir: Directory to store models and registry
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.model_dir / "registry.json"
        if not self.registry_path.exists():
            self._write_registry({})
        logger.info(f"ModelRegistry initialized | Path: {self.model_dir}")
    
    def _read_registry(self) -> Dict:
        """Read registry from disk."""
        try:
            return json.loads(self.registry_path.read_text())
        except Exception as e:
            logger.warning(f"Error reading registry: {e}")
            return {}
    
    def _write_registry(self, data: Dict) -> None:
        """Write registry to disk."""
        try:
            self.registry_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error writing registry: {e}")
    
    def register_model(self, strategy: str, artifact_path: str, metrics: Dict) -> None:
        """
        Register a model for a strategy.
        
        Args:
            strategy: Strategy name (e.g., 'intraday_mean_reversion')
            artifact_path: Path to model artifact file
            metrics: Dictionary of model metrics
        """
        reg = self._read_registry()
        reg[strategy] = {
            "artifact": artifact_path,
            "metrics": metrics,
            "registered_at": str(Path(artifact_path).stat().st_mtime) if Path(artifact_path).exists() else ""
        }
        self._write_registry(reg)
        logger.info(f"Model registered | Strategy: {strategy}, Artifact: {artifact_path}")
    
    def get_latest_model(self, strategy: str) -> Optional[str]:
        """
        Get the latest promoted model artifact path for a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Path to model artifact or None
        """
        reg = self._read_registry()
        entry = reg.get(strategy)
        if entry:
            return entry.get("artifact")
        return None
    
    def get_metrics(self, strategy: str) -> Dict:
        """
        Get metrics for a strategy's registered model.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Dictionary of metrics or empty dict
        """
        reg = self._read_registry()
        entry = reg.get(strategy, {})
        return entry.get("metrics", {})
    
    def list_models(self) -> Dict[str, Dict]:
        """
        List all registered models.
        
        Returns:
            Dictionary mapping strategy name to model info
        """
        return self._read_registry()
