"""
GPU Acceleration Benchmark Script

This script benchmarks CPU vs GPU aggregation performance for different
defense methods and model sizes.
"""

import time
from typing import Dict, List

import torch
import torch.nn as nn

# Import aggregation functions
import sys
sys.path.insert(0, '/home/ubuntu/Mutiple-FL')

from multi_server_fl.utils import (
    geometric_median_state_dicts,
    geometric_median_state_dicts_gpu,
    krum_aggregate,
    krum_aggregate_gpu,
    median_state_dicts,
    median_state_dicts_gpu,
    flatten_state_dict,
    unflatten_state_dict,
)


def create_dummy_model(num_params: int) -> nn.Module:
    """Create a simple model with specified number of parameters."""
    layer_size = int(num_params ** 0.5)
    return nn.Sequential(
        nn.Linear(layer_size, layer_size),
        nn.ReLU(),
        nn.Linear(layer_size, 10),
    )


def generate_test_states(
    model: nn.Module, num_clients: int, device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """Generate dummy client states for benchmarking."""
    states = []
    for _ in range(num_clients):
        state = {}
        for name, param in model.state_dict().items():
            # Add some random noise
            noisy_param = param + torch.randn_like(param) * 0.01
            state[name] = noisy_param.to(device)
        states.append(state)
    return states


def benchmark_geometric_median(
    num_clients: int, num_params: int, num_trials: int = 3
) -> Dict[str, float]:
    """Benchmark geometric median aggregation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking Geometric Median")
    print(f"Clients: {num_clients}, Parameters: {num_params:,}")
    print(f"{'='*70}")

    model = create_dummy_model(num_params)
    weights = [1.0] * num_clients

    # CPU benchmark
    cpu_states = generate_test_states(model, num_clients, torch.device("cpu"))
    cpu_times = []
    for trial in range(num_trials):
        start = time.time()
        _ = geometric_median_state_dicts(cpu_states, weights=weights)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU Trial {trial + 1}: {cpu_time:.3f}s")

    cpu_avg = sum(cpu_times) / len(cpu_times)

    # GPU benchmark
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_states = generate_test_states(model, num_clients, device)

        # Warmup
        _ = geometric_median_state_dicts_gpu(gpu_states, weights=weights, device=device)
        torch.cuda.synchronize()

        gpu_times = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            start = time.time()
            _ = geometric_median_state_dicts_gpu(gpu_states, weights=weights, device=device)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)
            print(f"  GPU Trial {trial + 1}: {gpu_time:.3f}s")

        gpu_avg = sum(gpu_times) / len(gpu_times)
        speedup = cpu_avg / gpu_avg

        print(f"\n  Results:")
        print(f"    CPU Average: {cpu_avg:.3f}s")
        print(f"    GPU Average: {gpu_avg:.3f}s")
        print(f"    Speedup: {speedup:.2f}x")

        return {
            "cpu_time": cpu_avg,
            "gpu_time": gpu_avg,
            "speedup": speedup,
        }
    else:
        print("  GPU not available, skipping GPU benchmark")
        return {"cpu_time": cpu_avg, "gpu_time": None, "speedup": None}


def benchmark_krum(
    num_clients: int, num_params: int, num_trials: int = 3
) -> Dict[str, float]:
    """Benchmark Krum aggregation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking Krum")
    print(f"Clients: {num_clients}, Parameters: {num_params:,}")
    print(f"{'='*70}")

    model = create_dummy_model(num_params)
    weights = [1.0] * num_clients
    num_malicious = max(1, num_clients // 5)

    # CPU benchmark
    cpu_states = generate_test_states(model, num_clients, torch.device("cpu"))
    cpu_times = []
    for trial in range(num_trials):
        start = time.time()
        _ = krum_aggregate(cpu_states, weights, num_malicious=num_malicious)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU Trial {trial + 1}: {cpu_time:.3f}s")

    cpu_avg = sum(cpu_times) / len(cpu_times)

    # GPU benchmark
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_states = generate_test_states(model, num_clients, device)

        # Warmup
        _ = krum_aggregate_gpu(gpu_states, weights, device=device, num_malicious=num_malicious)
        torch.cuda.synchronize()

        gpu_times = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            start = time.time()
            _ = krum_aggregate_gpu(gpu_states, weights, device=device, num_malicious=num_malicious)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)
            print(f"  GPU Trial {trial + 1}: {gpu_time:.3f}s")

        gpu_avg = sum(gpu_times) / len(gpu_times)
        speedup = cpu_avg / gpu_avg

        print(f"\n  Results:")
        print(f"    CPU Average: {cpu_avg:.3f}s")
        print(f"    GPU Average: {gpu_avg:.3f}s")
        print(f"    Speedup: {speedup:.2f}x")

        return {
            "cpu_time": cpu_avg,
            "gpu_time": gpu_avg,
            "speedup": speedup,
        }
    else:
        print("  GPU not available, skipping GPU benchmark")
        return {"cpu_time": cpu_avg, "gpu_time": None, "speedup": None}


def benchmark_median(
    num_clients: int, num_params: int, num_trials: int = 3
) -> Dict[str, float]:
    """Benchmark median aggregation."""
    print(f"\n{'='*70}")
    print(f"Benchmarking Median")
    print(f"Clients: {num_clients}, Parameters: {num_params:,}")
    print(f"{'='*70}")

    model = create_dummy_model(num_params)

    # CPU benchmark
    cpu_states = generate_test_states(model, num_clients, torch.device("cpu"))
    cpu_times = []
    for trial in range(num_trials):
        start = time.time()
        _ = median_state_dicts(cpu_states)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU Trial {trial + 1}: {cpu_time:.3f}s")

    cpu_avg = sum(cpu_times) / len(cpu_times)

    # GPU benchmark
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_states = generate_test_states(model, num_clients, device)

        # Warmup
        _ = median_state_dicts_gpu(gpu_states, device=device)
        torch.cuda.synchronize()

        gpu_times = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            start = time.time()
            _ = median_state_dicts_gpu(gpu_states, device=device)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            gpu_times.append(gpu_time)
            print(f"  GPU Trial {trial + 1}: {gpu_time:.3f}s")

        gpu_avg = sum(gpu_times) / len(gpu_times)
        speedup = cpu_avg / gpu_avg

        print(f"\n  Results:")
        print(f"    CPU Average: {cpu_avg:.3f}s")
        print(f"    GPU Average: {gpu_avg:.3f}s")
        print(f"    Speedup: {speedup:.2f}x")

        return {
            "cpu_time": cpu_avg,
            "gpu_time": gpu_avg,
            "speedup": speedup,
        }
    else:
        print("  GPU not available, skipping GPU benchmark")
        return {"cpu_time": cpu_avg, "gpu_time": None, "speedup": None}


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("GPU Acceleration Benchmark")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Test configurations
    configs = [
        # (num_clients, num_params, description)
        (10, 44_000, "Small (LeNet-like, 44K params)"),
        (50, 44_000, "Medium clients (50, LeNet)"),
        (100, 44_000, "Many clients (100, LeNet)"),
        (50, 11_000_000, "Large model (ResNet-like, 11M params)"),
    ]

    results = []

    for num_clients, num_params, desc in configs:
        print(f"\n{'#'*70}")
        print(f"Configuration: {desc}")
        print(f"{'#'*70}")

        # Geometric Median
        gm_result = benchmark_geometric_median(num_clients, num_params, num_trials=3)
        results.append({
            "config": desc,
            "method": "Geometric Median",
            **gm_result,
        })

        # Krum
        krum_result = benchmark_krum(num_clients, num_params, num_trials=3)
        results.append({
            "config": desc,
            "method": "Krum",
            **krum_result,
        })

        # Median
        median_result = benchmark_median(num_clients, num_params, num_trials=3)
        results.append({
            "config": desc,
            "method": "Median",
            **median_result,
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Configuration':<40} {'Method':<20} {'Speedup':>10}")
    print("-" * 70)
    for result in results:
        if result["speedup"] is not None:
            print(f"{result['config']:<40} {result['method']:<20} {result['speedup']:>9.2f}x")
        else:
            print(f"{result['config']:<40} {result['method']:<20} {'N/A':>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
