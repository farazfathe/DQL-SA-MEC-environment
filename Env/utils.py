from __future__ import annotations

from typing import Callable, Iterable
import math
import random


def free_space_path_loss_db(distance_m: float, freq_hz: float) -> float:
    """Compute free-space path loss in dB.

    FSPL(dB) = 20 log10(d) + 20 log10(f) + 32.44 (d in km, f in MHz)
    """
    if distance_m <= 0 or freq_hz <= 0:
        raise ValueError("distance_m and freq_hz must be positive")
    d_km = distance_m / 1000.0
    f_mhz = freq_hz / 1e6
    return 20 * math.log10(d_km) + 20 * math.log10(f_mhz) + 32.44


def constant_rate_task_generator(factory: Callable[[float, int], object],
                                 tasks_per_tick: int) -> Callable[[float], object]:
    """Create a simple generator that produces N tasks per tick using factory(time, i)."""
    def _gen(now: float):  # yields iterable of tasks
        return [factory(now, i) for i in range(tasks_per_tick)]

    return _gen


def poisson_task_generator(factory: Callable[[float, int], object], lambda_per_s: float) -> Callable[[float], Iterable[object]]:
    """Poisson process per second using thinning on 1s buckets for simplicity.

    Returns a generator that, when called with current time `now`, produces
    N ~ Poisson(lambda_per_s) tasks with timestamps within [now, now+1).
    """
    def _gen(now: float):
        n = _poisson(lambda_per_s)
        return [factory(now, i) for i in range(n)]

    return _gen


def _poisson(lmbda: float) -> int:
    # Knuth's algorithm
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(0, k - 1)


def energy_per_cycle_from_power(power_watts: float, cycles_per_second: float) -> float:
    """Compute energy per cycle (J) from average power and frequency.

    energy_per_cycle = power / f
    """
    if cycles_per_second <= 0:
        raise ValueError("cycles_per_second must be positive")
    return power_watts / float(cycles_per_second)


