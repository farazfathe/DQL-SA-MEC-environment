from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_latency(step_times: List[float], avg_latency: List[float]) -> None:
    plt.figure()
    plt.plot(step_times, avg_latency, label="Avg Latency (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (s)")
    plt.title("Average Task Latency over Time")
    plt.grid(True)
    plt.legend()


def plot_throughput(step_times: List[float], throughput: List[float]) -> None:
    plt.figure()
    plt.plot(step_times, throughput, label="Throughput (tasks/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Tasks per second")
    plt.title("Throughput over Time")
    plt.grid(True)
    plt.legend()


def plot_offloading(step_times: List[float], off_ratio: List[float], cloud_ratio: List[float]) -> None:
    plt.figure()
    plt.plot(step_times, off_ratio, label="Offloading Ratio")
    plt.plot(step_times, cloud_ratio, label="Cloud Offloading Ratio")
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio")
    plt.title("Offloading Ratios over Time")
    plt.grid(True)
    plt.legend()


def plot_energy(step_times: List[float], total_energy: List[float]) -> None:
    plt.figure()
    plt.plot(step_times, total_energy, label="Total Energy (J)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Cumulative Energy over Time")
    plt.grid(True)
    plt.legend()


def plot_sa_convergence(iters: List[int], objectives: List[float], temperatures: List[float] | None = None) -> None:
    plt.figure()
    plt.plot(iters, objectives, label="SA Objective")
    if temperatures is not None:
        ax2 = plt.twinx()
        ax2.plot(iters, temperatures, color="orange", alpha=0.6, label="Temperature")
        ax2.set_ylabel("Temperature")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("SA Convergence")
    plt.grid(True)
    plt.legend()


def plot_latency_hist(latencies: List[float]) -> None:
    plt.figure()
    if len(latencies) == 0:
        latencies = [0.0]
    plt.hist(latencies, bins=min(50, max(10, int(np.sqrt(len(latencies)) ))), log=True, edgecolor='black', alpha=0.7)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count (log)")
    plt.title("Latency Histogram")
    plt.grid(True, which='both')


def plot_latency_cdf(latencies: List[float]) -> None:
    plt.figure()
    if len(latencies) == 0:
        latencies = [0.0]
    data = np.sort(np.array(latencies))
    y = np.arange(1, len(data) + 1) / float(len(data))
    plt.step(data, y, where='post')
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("Latency CDF")
    plt.grid(True)


