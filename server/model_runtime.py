"""
Model runtime manager using llama.cpp for GGUF models with GPU acceleration.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator

try:
    from llama_cpp import Llama
except ImportError as e:
    raise RuntimeError(
        "Install llama-cpp-python: pip install llama-cpp-python\n"
        "For NVIDIA GPU (Windows): pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
    ) from e


@dataclass
class ModelSpec:
    """Specification for a single GGUF model."""
    alias: str
    gguf_path: str
    ctx_size: int = 1024
    gpu_layers: int = -1  # -1 = all layers on GPU


class ModelRegistry:
    """Registry and loader for GGUF models."""
    
    def __init__(self):
        self._specs: Dict[str, ModelSpec] = {}
        self._instances: Dict[str, Llama] = {}
        self._default_alias: Optional[str] = None

    def load_specs(self, specs: List[Dict[str, Any]], default_alias: Optional[str]):
        """
        Load model specifications from config.
        
        Args:
            specs: List of model specifications
            default_alias: Default model to use
        """
        for s in specs:
            alias = s["alias"]
            self._specs[alias] = ModelSpec(
                alias=alias,
                gguf_path=s["gguf_path"],
                ctx_size=int(s.get("ctx_size", 1024)),
                gpu_layers=int(s.get("gpu_layers", -1)),
            )
        self._default_alias = default_alias

    def list_aliases(self) -> List[str]:
        """Get list of available model aliases."""
        return list(self._specs.keys())

    def get_default_alias(self) -> Optional[str]:
        """Get the default model alias."""
        return self._default_alias

    def ensure_loaded(self, alias: Optional[str]) -> Llama:
        """
        Ensure a model is loaded and return it.
        
        Args:
            alias: Model alias to load (uses default if None)
            
        Returns:
            Loaded Llama model instance
            
        Raises:
            RuntimeError: If model alias not found or file doesn't exist
        """
        alias = alias or self._default_alias
        if not alias or alias not in self._specs:
            raise RuntimeError(f"Model alias not found: {alias}")
        
        # Return cached instance if already loaded
        if alias in self._instances:
            return self._instances[alias]
        
        # Load model
        spec = self._specs[alias]
        p = Path(spec.gguf_path)
        if not p.exists():
            raise RuntimeError(f"GGUF file not found: {p}")
        
        # Initialize with GPU layers for RTX 5070
        llm = Llama(
            model_path=str(p),
            n_ctx=spec.ctx_size,
            n_gpu_layers=spec.gpu_layers,
            verbose=False,  # Set to True for debugging
        )
        
        self._instances[alias] = llm
        return llm


class RuntimeOptions:
    """Runtime options manager with defaults."""
    
    def __init__(self, defaults: Dict[str, Any]):
        self.defaults = defaults or {}

    def merge(self, opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge runtime options with defaults.
        
        Args:
            opts: Options to merge with defaults
            
        Returns:
            Merged options dictionary
        """
        out = dict(self.defaults)
        if opts:
            out.update(opts)
        
        # Normalize keys and types
        out["temperature"] = float(out.get("temperature", 0.3))
        out["top_p"] = float(out.get("top_p", 0.9))
        out["top_k"] = int(out.get("top_k", 40))
        out["repeat_penalty"] = float(out.get("repeat_penalty", 1.1))
        out["num_ctx"] = int(out.get("num_ctx", 1024))
        out["num_predict"] = int(out.get("num_predict", 320))
        return out


class AIModelRuntime:
    """Main AI model runtime for chat and generation."""
    
    def __init__(self, registry: ModelRegistry, runtime_defaults: Dict[str, Any]):
        """
        Initialize AI model runtime.
        
        Args:
            registry: Model registry
            runtime_defaults: Default runtime options
        """
        self.registry = registry
        self.runtime_options = RuntimeOptions(runtime_defaults)

    @staticmethod
    def now() -> str:
        """Get current timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def chat(
        self, 
        alias: Optional[str], 
        messages: List[Dict[str, str]], 
        options: Optional[Dict[str, Any]], 
        stream: bool
    ) -> Any:
        """
        Run chat completion.
        
        Args:
            alias: Model alias to use
            messages: Chat messages
            options: Runtime options
            stream: Whether to stream response
            
        Returns:
            Full response text or generator for streaming
        """
        llm = self.registry.ensure_loaded(alias)
        opts = self.runtime_options.merge(options)
        
        args = dict(
            messages=messages,
            temperature=opts["temperature"],
            top_p=opts["top_p"],
            top_k=opts["top_k"],
            repeat_penalty=opts["repeat_penalty"],
            max_tokens=opts["num_predict"],
            stream=stream,
        )
        
        if stream:
            def gen() -> Generator[str, None, None]:
                for chunk in llm.create_chat_completion(**args):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
            return gen()
        else:
            out = llm.create_chat_completion(**args)
            return out.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

    def generate(
        self, 
        alias: Optional[str], 
        prompt: str, 
        options: Optional[Dict[str, Any]], 
        stream: bool
    ) -> Any:
        """
        Run text generation.
        
        Args:
            alias: Model alias to use
            prompt: Prompt text
            options: Runtime options
            stream: Whether to stream response
            
        Returns:
            Full response text or generator for streaming
        """
        llm = self.registry.ensure_loaded(alias)
        opts = self.runtime_options.merge(options)
        
        args = dict(
            prompt=prompt,
            temperature=opts["temperature"],
            top_p=opts["top_p"],
            top_k=opts["top_k"],
            repeat_penalty=opts["repeat_penalty"],
            max_tokens=opts["num_predict"],
            stream=stream,
        )
        
        if stream:
            def gen() -> Generator[str, None, None]:
                for chunk in llm.create_completion(**args):
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
            return gen()
        else:
            out = llm.create_completion(**args)
            return out.get("choices", [{}])[0].get("text", "") or ""
