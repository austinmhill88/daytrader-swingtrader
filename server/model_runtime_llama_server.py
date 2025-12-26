"""
Runtime against llama.cpp server OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions).
Use a CUDA-built llama.cpp server for GPU offload (n_gpu_layers -1).
"""
import time
import httpx
from typing import Dict, Any, List, Optional, Generator

class LlamaServerRegistry:
    def __init__(self, base_url: str, default_alias: str, model_map: Dict[str, str]):
        self.base_url = base_url.rstrip("/")
        self.default_alias = default_alias
        self.model_map = model_map

    def get_tag(self, alias: Optional[str]) -> str:
        return self.model_map.get(alias or self.default_alias) or (alias or self.default_alias)

class LlamaServerRuntime:
    def __init__(self, registry: LlamaServerRegistry, defaults: Dict[str, Any]):
        self.registry = registry
        self.defaults = defaults or {}

    @staticmethod
    def now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _opts(self, o: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        o = {**self.defaults, **(o or {})}
        return {
            "temperature": float(o.get("temperature", 0.3)),
            "top_p": float(o.get("top_p", 0.9)),
            "top_k": int(o.get("top_k", 40)),
            "repeat_penalty": float(o.get("repeat_penalty", 1.1)),
            "num_ctx": int(o.get("num_ctx", 2048)),
            "max_tokens": int(o.get("num_predict", 160)),
        }

    def chat(self, alias: Optional[str], messages: List[Dict[str, str]], options: Optional[Dict[str, Any]], stream: bool):
        tag = self.registry.get_tag(alias)
        payload = {
            "model": tag,
            "messages": messages,
            "stream": stream,
            **self._opts(options),
        }
        url = f"{self.registry.base_url}/v1/chat/completions"
        if stream:
            def gen() -> Generator[str, None, None]:
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                data = httpx.Response(status_code=200, content=line).json()
                            except Exception:
                                continue
                            delta = (data.get("choices", [{}])[0].get("delta") or {}).get("content", "")
                            if delta:
                                yield delta
            return gen()
        r = httpx.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return (r.json().get("choices", [{}])[0].get("message") or {}).get("content", "") or ""

    def generate(self, alias: Optional[str], prompt: str, options: Optional[Dict[str, Any]], stream: bool):
        tag = self.registry.get_tag(alias)
        payload = {
            "model": tag,
            "prompt": prompt,
            "stream": stream,
            **self._opts(options),
        }
        url = f"{self.registry.base_url}/v1/completions"
        if stream:
            def gen() -> Generator[str, None, None]:
                with httpx.Client(timeout=30) as client:
                    with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            try:
                                data = httpx.Response(status_code=200, content=line).json()
                            except Exception:
                                continue
                            piece = (data.get("choices", [{}])[0]).get("text", "")
                            if piece:
                                yield piece
            return gen()
        r = httpx.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return (r.json().get("choices", [{}])[0]).get("text", "") or ""