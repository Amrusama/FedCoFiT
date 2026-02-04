from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def deep_get(d: Mapping[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def deep_set(d: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: MutableMapping[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], MutableMapping):
            cur[p] = {}
        cur = cur[p]  # type: ignore[assignment]
    cur[parts[-1]] = value


def parse_value(s: str) -> Any:
    sl = s.lower()
    if sl in {"true", "false"}:
        return sl == "true"
    if sl in {"null", "none"}:
        return None
    try:
        if "." in s or "e" in sl:
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    # overrides as: --a.b.c value (handled by argparse) OR "a.b.c=value"
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        deep_set(cfg, k.strip(), parse_value(v.strip()))
    return cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def jsonl_append(path: str, obj: Mapping[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        gpus: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        info["gpus"] = gpus
    return info


def resolve_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


