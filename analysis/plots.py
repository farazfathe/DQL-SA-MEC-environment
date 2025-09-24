from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt


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


