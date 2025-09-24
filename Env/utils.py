from __future__ import annotations

from typing import Callable
import math


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


def energy_per_cycle_from_power(power_watts: float, cycles_per_second: float) -> float:
    """Compute energy per cycle (J) from average power and frequency.

    energy_per_cycle = power / f
    """
    if cycles_per_second <= 0:
        raise ValueError("cycles_per_second must be positive")
    return power_watts / float(cycles_per_second)


