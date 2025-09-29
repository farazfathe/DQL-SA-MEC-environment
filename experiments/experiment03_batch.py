from __future__ import annotations

import os
import sys
import json
import statistics as stats
from typing import List

import random

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # optional

try:
    import torch  # type: ignore
except Exception:
    torch = None  # optional


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.experiment03 import run_experiment_03


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def run_batch(
    seeds: List[int],
    sim_time_s: float = 300.0,
    edges_count: int = 5,
    cells_per_edge: int = 10,
    lambda_per_s: float = 0.5,
    # Edge-focused parameters
    edge_offload_bias: float = 0.8,
    cloud_penalty: float = 0.3,
    local_penalty: float = 0.1,
) -> str:
    results = []
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    for s in seeds:
        set_seed(s)
        summary_path = os.path.join(PROJECT_ROOT, "data", f"summary_exp03_seed{s}.json")
        run_experiment_03(
            sim_time_s=sim_time_s,
            edges_count=edges_count,
            cells_per_edge=cells_per_edge,
            lambda_per_s=lambda_per_s,
            edge_offload_bias=edge_offload_bias,
            cloud_penalty=cloud_penalty,
            local_penalty=local_penalty,
            summary_json_path=summary_path,
            pdf_path=os.path.join(PROJECT_ROOT, "data", f"Result_exp03_seed{s}.pdf"),
        )
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            pass

    if not results:
        return "{}"

    # Aggregate a few key metrics
    keys = [
        "success_rate",
        "offloading_ratio",
        "cloud_offloading_ratio",
        "avg_latency_s",
        "avg_wait_s",
        "total_energy_j",
        "energy_efficiency",
    ]
    agg = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        try:
            mu = stats.mean(vals)
            sd = stats.pstdev(vals)
        except Exception:
            mu, sd = 0.0, 0.0
        agg[k] = {"mean": mu, "std": sd}

    out_path = os.path.join(PROJECT_ROOT, "data", "summary_exp03_batch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds, 
            "aggregate": agg, 
            "runs": results,
            "edge_focused_params": {
                "edge_offload_bias": edge_offload_bias,
                "cloud_penalty": cloud_penalty,
                "local_penalty": local_penalty,
            }
        }, f, indent=2)
    return out_path


if __name__ == "__main__":
    # Test with different edge-focused parameter configurations
    configs = [
        {"edge_offload_bias": 0.6, "cloud_penalty": 0.2, "local_penalty": 0.05},
        {"edge_offload_bias": 0.8, "cloud_penalty": 0.3, "local_penalty": 0.1},
        {"edge_offload_bias": 1.0, "cloud_penalty": 0.5, "local_penalty": 0.15},
    ]
    
    for i, config in enumerate(configs):
        print(f"Running batch experiment with config {i+1}: {config}")
        path = run_batch(
            seeds=[0, 1, 2, 3, 4], 
            sim_time_s=300.0,
            edges_count=10,
            cells_per_edge=20,
            lambda_per_s=0.5,
            **config
        )
        print(f"Config {i+1} results saved to: {path}")
