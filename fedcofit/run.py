"""
FedCoFiT: Federated Co-Training for Large Language Models

Main entry point for running FedCoFiT experiments.

Usage:
    python -m fedcofit.run --config configs/config.yaml
    
    # Override parameters
    python -m fedcofit.run --config configs/config.yaml \\
        --override federated.rounds=5 \\
        --override experiment.seed=42
"""
from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from typing import Any, Dict

from fedcofit.utils import apply_overrides, load_config, set_global_seed
from fedcofit.orchestrator import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FedCoFiT: Federated Co-Training for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m fedcofit.run --config configs/config.yaml
  
  # Override parameters
  python -m fedcofit.run --config configs/config.yaml \\
    --override federated.rounds=5 \\
    --override hardware.num_clients=10
        """
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    p.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config key=value (repeatable), e.g. --override federated.rounds=5"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load config
    cfg: Dict[str, Any] = load_config(args.config)
    cfg = apply_overrides(cfg, args.override or [])
    
    # Ensure method name is set
    if "experiment" not in cfg:
        cfg["experiment"] = {}
    cfg["experiment"]["method"] = "FedCoFiT"
    
    # Set global seed
    set_global_seed(int(cfg["experiment"]["seed"]))
    
    # CUDA + multiprocessing safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Run experiment
    print(f"[FedCoFiT] Starting experiment with config: {args.config}")
    out_dir = run_experiment(cfg)
    print(f"[FedCoFiT] Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
