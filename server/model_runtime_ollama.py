"""
Runtime that proxies chat/generate to a local Ollama server (GPU-accelerated).
Assumes Ollama is installed and running (http://127.0.0.1:11434).
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Generator
import time
import httpx

@dataclass
class OllamaModelSpec:
    alias: str
    ollama_model: str  # e.g., "phi3-local"

class OllamaRegistry:
    def __init__(self, base_url: str):
        self._specs: Dict[str, OllamaModelSpec] = {}
        self._default_alias: Optional[str] = None
        self.base_url = base_url.rstrip("/")

    def load_specs(self, specs: List[Dict[str, Any]], default_alias: Optional[str]):
        for s in specs:
            alias = s["alias"]
            self._specs[alias] = OllamaModelSpec(alias=alias, ollama_model=s["ollama_model"])
        self._default_alias = default_alias

    def list_aliases(self) -> List[str]: return list(self._specs.keys())
    def get_default_alias(self) -> Optional[str]: return self._default_alias

    def get_model_tag(self, alias: Optional[str]) -> str:
        alias = alias or self._default_alias
        if not alias or alias not in self._specs:
            raise RuntimeError(f"Model alias not found: {alias}")
        return self._specs[alias].ollama_model

class RuntimeOptions:
    def __init__(self, defaults: Dict[str, Any]): self.defaults = defaults or {}
    def merge(self, opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(self.defaults)
        if opts: out.update(opts)
        return {
            "temperature": float(out.get("temperature", 0.3)),
            "top_p": float(out.get("top_p", 0.9)),
            "top_k": int(out.get("top_k", 40)),
            "repeat_penalty": float(out.get("repeat_penalty", 1.1)),
            "num_ctx": int(out.get("num_ctx", 2048)),
            "num_predict": int(out.get("num_predict", 160)),
        }

class OllamaRuntime:
    def __init__(self, registry: OllamaRegistry, runtime_defaults: Dict[str, Any]):
        self.registry = registry
        self.runtime_options = RuntimeOptions(runtime_defaults)
        self.base_url = registry.base_url

    @staticmethod
    def now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

    def chat(self, alias: Optional[str], messages: List[Dict[str, str]], options: Optional[Dict[str, Any]], stream: bool):
        tag = self.registry.get_model_tag(alias)
        payload = {"model": tag, "messages": messages, "options": self.runtime_options.merge(options), "stream": stream}
        if stream:
            def gen():
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line: continue
                            try: data = httpx.Response(status_code=200, content=line).json()
                            except Exception: continue
                            delta = data.get("message", {}).get("content", "")
                            if delta: yield delta
            return gen()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "") or ""

    def generate(self, alias: Optional[str], prompt: str, options: Optional[Dict[str, Any]], stream: bool):
        tag = self.registry.get_model_tag(alias)
        payload = {"model": tag, "prompt": prompt, "options": self.runtime_options.merge(options), "stream": stream}
        if stream:
            def gen():
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line: continue
                            try: data = httpx.Response(status_code=200, content=line).json()
                            except Exception: continue
                            piece = data.get("response", "")
                            if piece: yield piece
            return gen()
        r = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "") or ""