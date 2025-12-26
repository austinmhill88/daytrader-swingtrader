# AI Server Setup Guide

## Overview

The AI Runtime Server provides local AI model inference with Ollama-compatible API, web browsing capabilities, and controlled file system access. It runs entirely on your PC without external API dependencies.

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 5070 (or any CUDA-compatible GPU)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 5-10GB for models (depending on model size)
- **CPU**: Modern multi-core processor

### Software
- **OS**: Windows 10/11 (Linux also supported)
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or 12.1 (for NVIDIA GPU acceleration)
- **Visual Studio Build Tools**: Required for llama-cpp-python on Windows

## Installation

### 1. Install CUDA (for GPU acceleration)

Download and install CUDA Toolkit from NVIDIA:
- CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
- CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive

Verify installation:
```bash
nvcc --version
```

### 2. Install llama-cpp-python with GPU support

For Windows with NVIDIA GPU (CUDA 12.1):
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

For CUDA 11.8:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

For CPU-only (not recommended):
```bash
pip install llama-cpp-python
```

### 3. Install other dependencies

```bash
pip install fastapi uvicorn httpx pyyaml loguru
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Model Setup

### 1. Download GGUF Models

You need GGUF format models (quantized for efficiency). Recommended sources:

**HuggingFace:**
- Qwen2.5 3B Instruct (Recommended): 
  - `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB)
  - https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

- Llama 3.2 3B Instruct:
  - `llama-3.2-3b-instruct-q4_k_m.gguf` (~2GB)
  - https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

**Model Size Guide:**
- Q4_K_M: Good balance of quality and speed (recommended for RTX 5070)
- Q5_K_M: Higher quality, slightly slower
- Q8_0: Best quality, slower, more VRAM

### 2. Organize Models

Create a models directory:
```
C:\models\
  ├── qwen2.5-3b-instruct-q4_k_m.gguf
  ├── llama-3.2-3b-instruct-q4_k_m.gguf
  └── ...
```

## Configuration

### 1. Setup Environment Variables

Create a `.env` file or set system environment variables:

```bash
# Models directory
MODELS_DIR=C:\models

# Trading app root (for file sandbox)
DAYTRADER_ROOT=C:\daytrader-swingtrader

# Optional: SearxNG for web search
SEARXNG_URL=http://localhost:8080
```

### 2. Configure AI Server

Edit `config/ai-coder.yaml`:

```yaml
# Model registry
models:
  - alias: "qwen2.5:3b-instruct"
    gguf_path: "C:/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    ctx_size: 2048
    gpu_layers: -1  # -1 = all layers on GPU

default_model: "qwen2.5:3b-instruct"

# Network access
allow_network: true

# File sandbox
root_dir: "C:/daytrader-swingtrader"
allow_write: false  # Read-only by default for safety

# Runtime defaults
runtime_defaults:
  temperature: 0.3
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  num_ctx: 2048
  num_predict: 320
```

### 3. Configure Trading System

Edit `config/config.yaml` to enable AI integration:

```yaml
ai_server:
  enabled: true
  url: "http://127.0.0.1:8000"
  model: "qwen2.5:3b-instruct"
  timeout: 30
  features:
    trade_analysis: true
    daily_summary: true
    risk_analysis: true
```

## Running the AI Server

### Standalone Mode

Start the AI server independently:
```bash
python -m server.main
```

The server will be available at `http://127.0.0.1:8000`

### With Trading System

Start everything together:
```bash
python launcher.py --with-ai
```

### Verify Server is Running

Check health endpoint:
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": 0,
  "models_available": 1,
  "timestamp": "2024-01-15 10:30:00"
}
```

## API Endpoints

### Model Inference

**GET /api/version**
```bash
curl http://127.0.0.1:8000/api/version
```

**GET /api/tags** (list models)
```bash
curl http://127.0.0.1:8000/api/tags
```

**POST /api/chat** (chat completion)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct",
    "messages": [
      {"role": "user", "content": "What is a good trading strategy?"}
    ],
    "options": {"temperature": 0.3, "num_predict": 200}
  }'
```

**POST /api/generate** (text completion)
```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b-instruct",
    "prompt": "Explain risk management in trading:",
    "options": {"num_predict": 150}
  }'
```

### Web Access

**GET /web/fetch** (fetch web page)
```bash
curl "http://127.0.0.1:8000/web/fetch?url=https://example.com"
```

**GET /web/search** (search via SearxNG)
```bash
curl "http://127.0.0.1:8000/web/search?q=stock+market+news"
```

### File System

**GET /fs/list** (list directory)
```bash
curl "http://127.0.0.1:8000/fs/list?path=logs"
```

**GET /fs/read** (read file)
```bash
curl "http://127.0.0.1:8000/fs/read?path=logs/runtime.log&max_bytes=5000"
```

**GET /fs/tail** (tail file)
```bash
curl "http://127.0.0.1:8000/fs/tail?path=logs/trades.log&lines=50"
```

**POST /fs/diff** (preview changes)
```bash
curl -X POST http://127.0.0.1:8000/fs/diff \
  -H "Content-Type: application/json" \
  -d '{
    "path": "config/config.yaml",
    "new_content": "..."
  }'
```

## Performance Tuning

### GPU Optimization

For RTX 5070 with 12GB VRAM:
- Use `gpu_layers: -1` to offload all layers to GPU
- Recommended context size: 2048-4096 tokens
- Q4_K_M quantization provides best speed/quality balance

### Context Window

Adjust `num_ctx` based on your needs:
- 1024: Fast, good for short queries
- 2048: Balanced (recommended)
- 4096: Longer context, slower

### Generation Length

Adjust `num_predict`:
- 150: Quick responses
- 320: Standard responses (default)
- 500+: Detailed analysis

## Troubleshooting

### Model Not Loading

1. Check GGUF file path in config
2. Verify file exists and is not corrupted
3. Check VRAM availability (use `nvidia-smi`)

### CUDA Errors

1. Verify CUDA installation: `nvcc --version`
2. Check GPU drivers are up to date
3. Ensure llama-cpp-python is built with CUDA support

### Slow Performance

1. Verify GPU is being used (check `nvidia-smi` while running)
2. Reduce context size (`num_ctx`)
3. Use more aggressive quantization (Q4_K_M or lower)
4. Close other GPU-intensive applications

### Out of Memory

1. Reduce `num_ctx` (context window size)
2. Use smaller model or more aggressive quantization
3. Reduce `gpu_layers` (offload fewer layers to GPU)

## Security Considerations

### File Sandbox

- File access is restricted to `root_dir`
- Denied patterns take precedence over allowed
- Write access disabled by default
- Path traversal attempts are blocked

### Network Access

- Can be disabled via `allow_network: false`
- Web fetch respects robots.txt (non-blocking)
- No external API keys required

### Best Practices

1. Keep `allow_write: false` unless necessary
2. Use specific allowed/denied globs
3. Run server on localhost only
4. Monitor logs for suspicious activity
5. Keep models and dependencies updated

## Logs

Server logs are written to:
- `logs/ai_tools.log` - All server activity
- Rotated at 10MB
- Retained for 30 days

Monitor logs:
```bash
tail -f logs/ai_tools.log
```

## Next Steps

1. Test endpoints with sample requests
2. Integrate with trading system
3. Configure file sandbox patterns
4. Optimize model selection and parameters
5. Setup GUI for easy interaction

## Support

For issues:
1. Check logs: `logs/ai_tools.log`
2. Verify configuration: `config/ai-coder.yaml`
3. Test health endpoint: `http://127.0.0.1:8000/health`
4. Check GPU status: `nvidia-smi`
