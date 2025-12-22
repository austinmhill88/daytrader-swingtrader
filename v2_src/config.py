import yaml
import os

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve env vars like ${VAR}
    def resolve(val):
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            return os.environ.get(val[2:-1], "")
        return val
    def walk(d):
        if isinstance(d, dict):
            return {k: walk(resolve(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [walk(v) for v in d]
        return resolve(d)
    return walk(cfg)