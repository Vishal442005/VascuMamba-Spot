"""Benchmark O(T) VascuMamba-Spot vs O(T^2) Transformer latency and memory scaling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vascumamba_spot.models import VascuMambaConfig, VascuMambaSpot
from vascumamba_spot.visualization.plots import plot_scaling_curves


def measure_model(model: torch.nn.Module, seq_len: int, device: torch.device) -> tuple[float, float]:
    """Return (latency_ms, max_op_alloc_mb) for a single forward pass at sequence length T."""
    model.eval()
    xk = torch.randn(1, seq_len, 144, device=device)
    xp = torch.randn(1, seq_len, 36, device=device)

    with torch.no_grad():
        if device.type == "cuda":
            # Warm up CUDA / Triton JIT compilation before timing
            _ = model(xk, xp)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(xk, xp)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        else:
            t0 = time.perf_counter()
            _ = model(xk, xp)
            ms = (time.perf_counter() - t0) * 1000.0
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as prof:
                _ = model(xk, xp)
            max_op_bytes = max((evt.cpu_memory_usage for evt in prof.key_averages()), default=0)
            mem_mb = max_op_bytes / (1024.0 * 1024.0)
    return ms, mem_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark O(T) vs O(T^2) scaling.")
    parser.add_argument("--lengths", nargs="+", type=int, default=[250, 500, 1000, 2000, 4000])
    parser.add_argument("--out-json", type=Path, default=Path("outputs/scaling_benchmark.json"))
    parser.add_argument("--out-fig", type=Path, default=Path("paper/figures/fig4_scaling.pdf"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mamba = VascuMambaSpot(VascuMambaConfig(backbone_type="vascumamba")).to(device)
    trans = VascuMambaSpot(VascuMambaConfig(backbone_type="transformer")).to(device)

    m_ms, m_mem, t_ms, t_mem = [], [], [], []
    for length in args.lengths:
        ms1, mem1 = measure_model(mamba, length, device)
        ms2, mem2 = measure_model(trans, length, device)
        m_ms.append(ms1)
        m_mem.append(mem1)
        t_ms.append(ms2)
        t_mem.append(mem2)
        print(f"T={length:5d} | Mamba: {ms1:7.1f}ms ({mem1:6.1f}MB) | Trans: {ms2:7.1f}ms ({mem2:6.1f}MB)")

    payload = {
        "lengths": args.lengths,
        "vascumamba_params": mamba.count_parameters(),
        "transformer_params": trans.count_parameters(),
        "mamba_ms": m_ms,
        "mamba_mem_mb": m_mem,
        "transformer_ms": t_ms,
        "transformer_mem_mb": t_mem,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_scaling_curves(args.lengths, m_ms, t_ms, m_mem, t_mem, args.out_fig)


if __name__ == "__main__":
    main()
