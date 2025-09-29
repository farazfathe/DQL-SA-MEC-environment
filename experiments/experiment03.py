from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import random

import simpy
import yaml
import numpy as np

# Ensure project root is importable when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use non-interactive backend for headless PDF generation
import matplotlib
matplotlib.use("Agg")

from Env import Cell, EdgeServer, Cloud
from Env.task import Task
from Env.utils import energy_per_cycle_from_power
from Env.utils import poisson_task_generator

from algos.DQN_agent import DQNAgent
from algos.dqn_utils import compute_reward
from algos.simulated_annealing import SimulatedAnnealing
from algos.QL_SA_wrapper import QLSAWrapper

from analysis.Metrics import MetricsLogger
from analysis.plots import (
    plot_latency,
    plot_throughput,
    plot_offloading,
    plot_energy,
    plot_latency_hist,
    plot_latency_cdf,
    plot_scheduling_time,
    plot_accept_failure,
    plot_load_balance,
    plot_energy_efficiency,
    plot_rl,
)

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_env_config(cfg: Dict) -> Dict:
    # Helpers to be robust to YAML variations
    def as_map(x):
        return x if isinstance(x, dict) else {}

    def parse_cpu_rate_hz(value) -> float:
        # Accept numeric (GHz) or string like "8 GHz"
        try:
            if isinstance(value, (int, float)):
                # Interpret as GHz
                return float(value) * 1e9
            s = str(value).strip().lower()
            # Extract leading float
            import re
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
            num = float(m.group(1)) if m else 1.0
            return num * (1e9 if "ghz" in s or "g" in s else (1e6 if "mhz" in s or "m" in s else 1.0))
        except Exception:
            return 1e9

    cell_cfg = as_map(cfg.get("cell_devices", {}))
    edge_cfg = as_map(cfg.get("edge_devices", {}))
    cloud_cfg = as_map(cfg.get("cloud", {}))
    net_cfg = as_map(cfg.get("network", {}))
    sim_cfg = as_map(cfg.get("simulation", {}))

    cell_specs = as_map(cell_cfg.get("specs", {}))
    edge_specs = as_map(edge_cfg.get("specs", {}))
    cloud_specs = as_map(cloud_cfg.get("specs", {}))

    # Parse bandwidths in bits/s
    bw = net_cfg.get("bandwidth", {})
    lat = as_map(net_cfg.get("latency", {}))

    def parse_rate(s: str) -> float:
        if not isinstance(s, str):
            return float(s)
        s = s.strip().lower()
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        num = float(m.group(1)) if m else 0.0
        if "gbps" in s or "g" in s:
            return num * 1e9
        if "mbps" in s or "m" in s:
            return num * 1e6
        if "kbps" in s or "k" in s:
            return num * 1e3
        return num

    def parse_time(s: str) -> float:
        # Return seconds from strings like "10 ms" or numeric seconds
        try:
            if isinstance(s, (int, float)):
                return float(s)
            ss = str(s).strip().lower()
            import re
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", ss)
            num = float(m.group(1)) if m else 0.0
            if "ms" in ss:
                return num / 1000.0
            if "s" in ss:
                return num
            return num
        except Exception:
            return 0.0

    return {
        "cell": {
            "cpu_rate": parse_cpu_rate_hz(cell_specs.get("cpu", "2 GHz")),
            "battery_j": 500.0,
            "tx_energy_per_bit": 1e-9,
        },
        "edge": {
            "cpu_rate": parse_cpu_rate_hz(edge_specs.get("cpu", "8 GHz")),
            "energy_j": 1e6,
            "compute_energy_per_cycle": 5e-12,
        },
        "cloud": {
            "cpu_rate": parse_cpu_rate_hz(cloud_specs.get("cpu", "16 GHz")),
            "energy_j": 1e9,
            "compute_energy_per_cycle": 5e-12,
        },
        "network": {
            "iot_to_uav": parse_rate(bw.get("iot_to_uav", "100 Mbps")),
            "uav_to_mec": parse_rate(bw.get("uav_to_mec", "1 Gbps")),
            "iot_to_mec": parse_rate(bw.get("iot_to_mec", "50 Mbps")),
            "latency": {
                "iot_to_uav": parse_time(lat.get("iot_to_uav", "10 ms")),
                "uav_to_mec": parse_time(lat.get("uav_to_mec", "5 ms")),
                "iot_to_mec": parse_time(lat.get("iot_to_mec", "25 ms")),
            },
            "jitter_s": float(as_map(net_cfg).get("jitter_ms", 0.0)) / 1000.0,
        },
        "simulation": {
            "duration": float(sim_cfg.get("duration", 3600)),
        },
    }


def parse_algo_config(cfg: Dict) -> Tuple[Dict, Dict]:
    ql_cfg = cfg.get("q_learning", {})
    sa_cfg = cfg.get("simulated_annealing", {})
    return ql_cfg, sa_cfg


def make_task_factory(bit_size_bits: int, memory_bytes: int, cpu_cycles: int, deadline_s: float):
    def _factory(now: float, i: int) -> Task:
        return Task(
            task_id=f"t_{int(now)}_{i}",
            bit_size=bit_size_bits,
            memory_required=memory_bytes,
            cpu_cycles=cpu_cycles,
            deadline=now + deadline_s,
            created_at=now,
        )
    return _factory


def run_experiment_03(
    sim_time_s: Optional[float] = None,
    env_cfg_path: str = os.path.join("data", "env.yaml"),
    algo_cfg_path: str = os.path.join("data", "algo.yaml"),
    pdf_path: str = os.path.join("data", "Result_exp03.pdf"),
    edges_count: int = 200,
    cells_per_edge: int = 20,
    lambda_per_s: float = 0.005,
    # Failure probabilities
    failure_prob_local: float = 0.02,
    failure_prob_edge: float = 0.02,
    failure_prob_cloud: float = 0.01,
    # Contention and energy variability
    edge_contention_coeff: float = 0.05,
    edge_energy_var_coeff: float = 0.2,
    cloud_contention_coeff: float = 0.02,
    cloud_energy_var_coeff: float = 0.1,
    summary_json_path: Optional[str] = None,
    timeseries_csv_path: Optional[str] = None,
    aggregated_generation: bool = True,
    # DQN training knobs integrated into the main workflow
    warmup_windows: int = 5,
    eval_interval_windows: int = 10,
    # Edge-focused parameters
    edge_offload_bias: float = 0.8,  # Higher bias towards edge offloading
    cloud_penalty: float = 0.3,  # Penalty for cloud offloading
    local_penalty: float = 0.1,  # Small penalty for local execution
) -> str:
    # Load configs
    env_raw = load_yaml(env_cfg_path)
    algo_raw = load_yaml(algo_cfg_path)
    env_conf = parse_env_config(env_raw)
    ql_conf, sa_conf = parse_algo_config(algo_raw)

    # SimPy env
    env = simpy.Environment()

    # Device capacities from env.yaml
    cell_cpu_rate = float(env_conf["cell"]["cpu_rate"])  # cycles/s
    cell_power_w = 2.0
    cell_energy_per_cycle = energy_per_cycle_from_power(cell_power_w, cell_cpu_rate)
    cell_battery_j = float(env_conf["cell"]["battery_j"])

    edge_cpu_rate = float(env_conf["edge"]["cpu_rate"])  # cycles/s
    edge_energy_j = float(env_conf["edge"]["energy_j"])
    edge_compute_energy_per_cycle = float(env_conf["edge"]["compute_energy_per_cycle"]) 

    cloud_cpu_rate = float(env_conf["cloud"]["cpu_rate"])  # cycles/s
    cloud_energy_j = float(env_conf["cloud"]["energy_j"])
    cloud_compute_energy_per_cycle = float(env_conf["cloud"]["compute_energy_per_cycle"]) 

    # Network
    bw_iot_to_mec = float(env_conf["network"]["iot_to_mec"])  # bits/s (uplink)

    # Build infrastructure: 1 cloud, N edges (parameterized)
    edges: List[EdgeServer] = []
    for i in range(int(edges_count)):
        e = EdgeServer(
            edge_id=f"edge-{i}",
            cpu_rate_cycles_per_s=edge_cpu_rate,
            energy_joules=edge_energy_j,
            compute_energy_j_per_cycle=edge_compute_energy_per_cycle,
        )
        e.contention_coeff = float(edge_contention_coeff)
        e.energy_variability_coeff = float(edge_energy_var_coeff)
        e.failure_prob = float(failure_prob_edge)
        e.service_time_jitter_coeff = 0.1
        e.start(env)
        edges.append(e)

    cloud = Cloud(
        cpu_rate_cycles_per_s=cloud_cpu_rate,
        energy_joules=cloud_energy_j,
        compute_energy_j_per_cycle=cloud_compute_energy_per_cycle,
    )
    cloud.contention_coeff = float(cloud_contention_coeff)
    cloud.energy_variability_coeff = float(cloud_energy_var_coeff)
    cloud.failure_prob = float(failure_prob_cloud)
    cloud.service_time_jitter_coeff = 0.1
    cloud.start(env)

    # Metrics hooks for edge/cloud
    metrics = MetricsLogger(record_events=False)

    def on_edge_done(task: Task, energy_cost: float):
        metrics.mark_completed(task.task_id, float(env.now), energy_cost)

    def on_cloud_done(task: Task, energy_cost: float):
        metrics.mark_completed(task.task_id, float(env.now), energy_cost)

    for e in edges:
        e.on_task_completed = on_edge_done
        # Event hooks for detailed lifecycle logging
        e.on_task_enqueued = (lambda task, time_s, eid=e.edge_id: metrics.log_event(task.task_id, float(time_s), "edge_enqueue", node=eid))
        e.on_task_started = (lambda task, eid=e.edge_id: metrics.log_event(task.task_id, float(env.now), "edge_start", node=eid))
        e.on_task_dropped = (lambda task, eid=e.edge_id: metrics.log_event(task.task_id, float(env.now), "edge_drop", node=eid))
    cloud.on_task_completed = on_cloud_done
    cloud.on_task_enqueued = (lambda task, time_s: metrics.log_event(task.task_id, float(time_s), "cloud_enqueue", node="cloud"))
    cloud.on_task_started = (lambda task: metrics.log_event(task.task_id, float(env.now), "cloud_start", node="cloud"))
    cloud.on_task_dropped = (lambda task: metrics.log_event(task.task_id, float(env.now), "cloud_drop", node="cloud"))

    # Task generation profile
    bit_size_bits = 1_000_000  # 1 Mb per task
    memory_bytes = 64 * 1024 * 1024  # 64 MB
    cpu_cycles = int(5e8)  # 0.5 Gcycles
    deadline_s = 1.0

    # cells per edge => total cells is parameterized
    cells: List[Cell] = []
    for i, e in enumerate(edges):
        for j in range(int(cells_per_edge)):
            cell_idx = i * int(cells_per_edge) + j
            cell_id = f"cell-{cell_idx}"
            # Per-cell Poisson arrivals with unique task IDs incorporating cell_id
            gen = poisson_task_generator(
                factory=lambda now, k, cid=cell_id: Task(
                    task_id=f"{cid}_t_{int(now*1000)}_{k}",
                    bit_size=bit_size_bits,
                    memory_required=memory_bytes,
                    cpu_cycles=cpu_cycles,
                    deadline=now + deadline_s,
                    created_at=now,
                ),
                lambda_per_s=float(lambda_per_s),
            )
            c = Cell(
                cell_id=cell_id,
                cpu_rate_cycles_per_s=cell_cpu_rate,
                battery_energy_joules=cell_battery_j,
                task_generator=gen,
                edge_id=e.edge_id,
            )
            c.compute_energy_j_per_cycle = cell_energy_per_cycle
            c.tx_energy_j_per_bit_to_edge = float(env_conf["cell"]["tx_energy_per_bit"]) 
            c.failure_prob_local = float(failure_prob_local)
            c.energy_variability_coeff_local = 0.1
            cells.append(c)

    # Start cells only if not using aggregated generation
    if not aggregated_generation:
        for c in cells:
            c.start(env)

    # --- Policy (DQL + SA) from algo.yaml ---
    ql = DQNAgent(
        alpha=float(ql_conf.get("alpha", 0.001)),
        gamma=float(ql_conf.get("gamma", 0.9)),
        epsilon=float(ql_conf.get("epsilon", 0.2)),
    )
    sa = SimulatedAnnealing(
        initial_temperature=float(sa_conf.get("initial_temperature", 1.0)),
        cooling_rate=float(sa_conf.get("cooling_rate", 0.99)),
        min_temperature=float(sa_conf.get("min_temperature", 1e-3)),
        steps_per_call=int(sa_conf.get("steps_per_call", 20)),
    )
    policy = QLSAWrapper(ql=ql, sa=sa)
    
    # DQN training backlog (shared with driver)
    training_backlog = []
    
    def driver():
        nonlocal training_backlog
        window_s = 1.0
        iteration = 0
        last_objective = 0.0
        
        # DQN training episode tracking
        training_episodes = 0
        max_training_episodes = 10  # Run exactly 10 training iterations
        
        # events tracking disabled to reduce memory; failure ratio approximated as 0
        while True:
            completed_this_tick = 0
            offloaded_count = 0
            cloud_offloaded_count = 0
            latencies: List[float] = []
            waits: List[float] = []
            energy_this_tick = 0.0
            sched_time_this_tick = 0.0
            sched_decisions_this_tick = 0
            arrivals_this_tick = 0

            # Generate and handle tasks per cell for this 1s window
            for c in cells:
                new_tasks: List[Task] = list(c.task_generator(float(env.now)))
                arrivals_this_tick += len(new_tasks)
                for j, t in enumerate(new_tasks):
                    t.source_cell_id = c.cell_id
                    # RL action set: local, cloud, or any edge
                    actions = ["local", "cloud"] + [("edge", e.edge_id) for e in edges]
                    
                    # Edge-focused score shaping function
                    def _edge_focused_shaping(act):
                        if act == "local":
                            return -local_penalty  # Small penalty for local execution
                        if act == "cloud":
                            return -cloud_penalty  # Higher penalty for cloud offloading
                        if isinstance(act, tuple) and act[0] == "edge":
                            return edge_offload_bias  # High reward for edge offloading
                        return 0.0
                    _t0 = time.perf_counter()
                    if iteration < int(warmup_windows):
                        # Edge-focused warm-up: try edge first, then local, else cloud
                        if any(isinstance(a, tuple) and a[0] == "edge" for a in actions):
                            # Prefer the assigned edge for this cell
                            choice = ("edge", c.edge_id) if c.edge_id is not None else next(a for a in actions if isinstance(a, tuple) and a[0] == "edge")
                        elif "local" in actions:
                            choice = "local"
                        else:
                            choice = "cloud"
                    else:
                        choice = policy.select_action(c.cell_id, actions, score_shaping=_edge_focused_shaping)
                    sched_time_this_tick += (time.perf_counter() - _t0)
                    sched_decisions_this_tick += 1

                    # Approximate immediate reward for DQL update
                    attempt = 1
                    offload_type = "local" if choice == "local" else ("cloud" if choice == "cloud" else "edge")
                    base_lat = float(env_conf["network"]["latency"]["iot_to_mec"]) if offload_type != "local" else 0.0
                    jitter = float(env_conf["network"].get("jitter_s", 0.0)) if offload_type != "local" else 0.0
                    tx_time = 0.0 if offload_type == "local" else ((t.bit_size / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0) + max(0.0, base_lat + random.uniform(-jitter, jitter)))
                    if offload_type == "local":
                        compute_time = t.estimated_compute_time(cell_cpu_rate)
                        energy = t.cpu_cycles * c.compute_energy_j_per_cycle
                        success_p = 1.0 - float(failure_prob_local)
                    elif offload_type == "cloud":
                        compute_time = t.estimated_compute_time(cloud_cpu_rate)
                        energy = (t.bit_size * float(env_conf["cell"]["tx_energy_per_bit"])) + (t.cpu_cycles * cloud_compute_energy_per_cycle)
                        success_p = 1.0 - float(failure_prob_cloud)
                    else:
                        compute_time = t.estimated_compute_time(edge_cpu_rate)
                        energy = (t.bit_size * float(env_conf["cell"]["tx_energy_per_bit"])) + (t.cpu_cycles * edge_compute_energy_per_cycle)
                        success_p = 1.0 - float(failure_prob_edge)
                    latency_est = tx_time + compute_time
                    will_meet_deadline = (float(env.now) + latency_est) <= t.deadline
                    success_flag = will_meet_deadline and (random.random() < success_p)
                    r = compute_reward({
                        "success": bool(success_flag),
                        "latency": float(latency_est),
                        "energy": float(energy),
                        "attempt": int(attempt),
                        "offload_type": offload_type,
                    }, 0.0)
                    
                    # Apply edge-focused reward shaping
                    if offload_type == "edge":
                        r += edge_offload_bias  # Bonus for edge offloading
                    elif offload_type == "cloud":
                        r -= cloud_penalty  # Penalty for cloud offloading
                    elif offload_type == "local":
                        r -= local_penalty  # Small penalty for local execution
                    try:
                        policy.update(
                            state=c.cell_id,
                            action=choice,
                            reward=float(r),
                            next_state=c.cell_id,
                            next_actions=actions,
                            done=True,
                        )
                    except TypeError:
                        ql.update(
                            state=c.cell_id,
                            action=choice,
                            reward=float(r),
                            next_state=c.cell_id,
                            next_actions=actions,
                            done=True,
                        )

                    now = float(env.now)
                    if choice == "local":
                        if c.execute_locally(t, now):
                            if t.started_at is not None:
                                metrics.log_event(t.task_id, float(t.started_at), "local_start", node=c.cell_id)
                            if t.finished_at is not None:
                                metrics.log_event(t.task_id, float(t.finished_at), "local_complete", node=c.cell_id)
                            completed_this_tick += 1
                            latencies.append(t.finished_at - t.created_at)
                            if t.started_at is not None:
                                waits.append((t.started_at - (t.queued_at or t.created_at)) if t.queued_at is not None else 0.0)
                            energy_local = t.cpu_cycles * c.compute_energy_j_per_cycle
                            energy_this_tick += energy_local
                            metrics.log_task(
                                task_id=t.task_id,
                                source_cell_id=t.source_cell_id,
                                created_at=t.created_at,
                                started_at=t.started_at,
                                finished_at=t.finished_at,
                                queued_at=(t.queued_at if t.queued_at is not None else None),
                                deadline=t.deadline,
                                was_offloaded=False,
                                was_to_cloud=False,
                                energy_j=energy_local,
                            )
                        else:
                            choice = ("edge", c.edge_id) if c.edge_id is not None else ("cloud",)

                    if isinstance(choice, tuple) and choice[0] == "edge":
                        base_lat2 = float(env_conf["network"]["latency"]["iot_to_mec"])
                        jitter2 = float(env_conf["network"].get("jitter_s", 0.0))
                        tx_time2 = (t.bit_size / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0) + max(0.0, base_lat2 + random.uniform(-jitter2, jitter2))
                        cost = c.prepare_offload_to_edge(t)
                        if cost is not None:
                            metrics.log_event(t.task_id, float(env.now), "tx_start", node=c.cell_id)
                            def _send_to_edge(task=t, edge_id=choice[1], energy_cost=cost, tx_delay=tx_time2):
                                yield env.timeout(tx_delay)
                                metrics.log_event(task.task_id, float(env.now), "tx_complete", node=edge_id)
                                idx = int(edge_id.split("-")[-1])
                                edges[idx].put(task)
                                metrics.log_task(
                                    task_id=task.task_id,
                                    source_cell_id=task.source_cell_id,
                                    created_at=task.created_at,
                                    started_at=None,
                                    finished_at=None,
                                    queued_at=(task.queued_at if task.queued_at is not None else None),
                                    deadline=task.deadline,
                                    was_offloaded=True,
                                    was_to_cloud=False,
                                    energy_j=energy_cost,
                                )
                            offloaded_count += 1
                            energy_this_tick += cost
                            env.process(_send_to_edge())
                    elif choice == "cloud":
                        base_lat2 = float(env_conf["network"]["latency"]["iot_to_mec"]) 
                        jitter2 = float(env_conf["network"].get("jitter_s", 0.0))
                        tx_time2 = (t.bit_size / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0) + max(0.0, base_lat2 + random.uniform(-jitter2, jitter2))
                        cost = c.prepare_offload_to_edge(t)
                        if cost is not None:
                            metrics.log_event(t.task_id, float(env.now), "tx_start", node=c.cell_id)
                            def _send_to_cloud(task=t, energy_cost=cost, tx_delay=tx_time2):
                                yield env.timeout(tx_delay)
                                metrics.log_event(task.task_id, float(env.now), "tx_complete", node="cloud")
                                cloud.put(task)
                                metrics.log_task(
                                    task_id=task.task_id,
                                    source_cell_id=task.source_cell_id,
                                    created_at=task.created_at,
                                    started_at=None,
                                    finished_at=None,
                                    queued_at=(task.queued_at if task.queued_at is not None else None),
                                    deadline=task.deadline,
                                    was_offloaded=True,
                                    was_to_cloud=True,
                                    energy_j=energy_cost,
                                )
                            cloud_offloaded_count += 1
                            energy_this_tick += cost
                            env.process(_send_to_cloud())

            # Advance time window
            yield env.timeout(window_s)

            # Utilization/energy snapshot for this window
            node_util: Dict[str, float] = {}
            node_q: Dict[str, int] = {}
            node_energy: Dict[str, float] = {}
            total_busy_s = 0.0
            for e in edges:
                st = e.pop_window_counters()
                node_util[e.edge_id] = (st["busy_time_s"] / window_s) if window_s > 0 else 0.0
                node_q[e.edge_id] = st["completed"]
                node_energy[e.edge_id] = st["energy_j"]
                total_busy_s += float(st["busy_time_s"])
            cst = cloud.pop_window_counters()
            node_util["cloud"] = (cst["busy_time_s"] / window_s) if window_s > 0 else 0.0
            node_q["cloud"] = cst["completed"]
            node_energy["cloud"] = cst["energy_j"]
            total_busy_s += float(cst["busy_time_s"])

            completed_nodes = sum(node_q.values())
            completed_this_window = completed_this_tick + completed_nodes

            # Approximate failures (events disabled): set to 0 for this window
            failures_this_window = 0

            # Load balance CV across edges
            edge_utils = [u for k, u in node_util.items() if k != "cloud"]
            if edge_utils:
                mu = float(sum(edge_utils) / len(edge_utils))
                if mu > 0:
                    var = float(sum((x - mu) ** 2 for x in edge_utils) / len(edge_utils))
                    lb_cv = (var ** 0.5) / mu
                else:
                    lb_cv = 0.0
            else:
                lb_cv = 0.0

            

            # RL convergence logging with edge-focused metrics
            avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
            total_new = max(1, completed_this_tick + offloaded_count)
            edge_offload_ratio = offloaded_count / total_new
            cloud_offload_ratio = cloud_offloaded_count / total_new if total_new > 0 else 0.0
            local_ratio = (completed_this_tick - offloaded_count) / total_new if total_new > 0 else 0.0
            
            # Edge-focused objective function
            Z = 0.4 * avg_lat + 0.4 * (1.0 - edge_offload_ratio) + 0.1 * cloud_offload_ratio + 0.1 * (energy_this_tick / 1000.0)
            reward_window = -(Z - last_objective)
            last_objective = Z
            
            # Log step with proper metrics including busy time
            metrics.log_step(
                time_s=float(env.now),
                completed_tasks=completed_this_tick,
                offloaded=offloaded_count,
                cloud_offloaded=cloud_offloaded_count,
                energy_j=energy_this_tick,
                latencies=latencies,
                waits=waits,
                server_utilization=node_util,
                queue_lengths={k: v for k, v in node_q.items()},
                node_completed=node_q,
                node_energy_j=node_energy,
                window_s=window_s,
                sched_time_s=sched_time_this_tick,
                decisions=sched_decisions_this_tick,
                arrivals=arrivals_this_tick,
                acceptance_ratio=1.0 - (failures_this_window / max(1, arrivals_this_tick)) if arrivals_this_tick > 0 else None,
                failure_ratio=failures_this_window / max(1, arrivals_this_tick) if arrivals_this_tick > 0 else None,
                busy_time_total_s=total_busy_s,
                load_balance_cv=lb_cv,
            )
            
            try:
                metrics.log_rl(iteration=iteration, epsilon=float(ql.epsilon), reward=float(reward_window), objective=float(Z))
            except Exception:
                pass

            # DQN Training Episodes - trigger training every eval_interval_windows
            if iteration > 0 and iteration % eval_interval_windows == 0 and training_episodes < max_training_episodes:
                training_episodes += 1
                print(f"Starting DQN training episode {training_episodes} at iteration {iteration}")
                
                # Training episode data collection
                episode_data = {
                    "episode": training_episodes,
                    "iteration": iteration,
                    "training_steps": [],
                    "total_reward": 0.0,
                    "edge_actions": 0,
                    "cloud_actions": 0,
                    "local_actions": 0,
                    "successful_actions": 0,
                    "avg_latency": 0.0,
                    "avg_energy": 0.0,
                }
                
                # Run a focused training session
                training_steps = 100  # 100 training steps per episode
                episode_latencies = []
                episode_energies = []
                
                for step in range(training_steps):
                    # Sample a random cell for training
                    cell = random.choice(cells)
                    actions = ["local", "cloud"] + [("edge", e.edge_id) for e in edges]
                    
                    # Use current policy for action selection
                    choice = policy.select_action(cell.cell_id, actions, score_shaping=_edge_focused_shaping)
                    
                    # Simulate task execution and reward
                    offload_type = "local" if choice == "local" else ("cloud" if choice == "cloud" else "edge")
                    base_lat = float(env_conf["network"]["latency"]["iot_to_mec"]) if offload_type != "local" else 0.0
                    jitter = float(env_conf["network"].get("jitter_s", 0.0)) if offload_type != "local" else 0.0
                    tx_time = 0.0 if offload_type == "local" else ((bit_size_bits / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0) + max(0.0, base_lat + random.uniform(-jitter, jitter)))
                    
                    if offload_type == "local":
                        compute_time = cpu_cycles / cell_cpu_rate
                        energy = cpu_cycles * cell_energy_per_cycle
                        success_p = 1.0 - float(failure_prob_local)
                    elif offload_type == "cloud":
                        compute_time = cpu_cycles / cloud_cpu_rate
                        energy = (bit_size_bits * float(env_conf["cell"]["tx_energy_per_bit"])) + (cpu_cycles * cloud_compute_energy_per_cycle)
                        success_p = 1.0 - float(failure_prob_cloud)
                    else:
                        compute_time = cpu_cycles / edge_cpu_rate
                        energy = (bit_size_bits * float(env_conf["cell"]["tx_energy_per_bit"])) + (cpu_cycles * edge_compute_energy_per_cycle)
                        success_p = 1.0 - float(failure_prob_edge)
                    
                    latency_est = tx_time + compute_time
                    success_flag = random.random() < success_p
                    
                    r = compute_reward({
                        "success": bool(success_flag),
                        "latency": float(latency_est),
                        "energy": float(energy),
                        "attempt": 1,
                        "offload_type": offload_type,
                    }, 0.0)
                    
                    # Apply edge-focused reward shaping
                    if offload_type == "edge":
                        r += edge_offload_bias
                    elif offload_type == "cloud":
                        r -= cloud_penalty
                    elif offload_type == "local":
                        r -= local_penalty
                    
                    # Collect training step data
                    step_data = {
                        "step": step,
                        "cell_id": cell.cell_id,
                        "action": choice,
                        "offload_type": offload_type,
                        "reward": float(r),
                        "success": success_flag,
                        "latency": float(latency_est),
                        "energy": float(energy),
                        "epsilon": float(ql.epsilon),
                    }
                    episode_data["training_steps"].append(step_data)
                    episode_data["total_reward"] += float(r)
                    
                    # Count action types
                    if offload_type == "edge":
                        episode_data["edge_actions"] += 1
                    elif offload_type == "cloud":
                        episode_data["cloud_actions"] += 1
                    else:
                        episode_data["local_actions"] += 1
                    
                    if success_flag:
                        episode_data["successful_actions"] += 1
                        episode_latencies.append(float(latency_est))
                        episode_energies.append(float(energy))
                    
                    # Update policy
                    try:
                        policy.update(
                            state=cell.cell_id,
                            action=choice,
                            reward=float(r),
                            next_state=cell.cell_id,
                            next_actions=actions,
                            done=True,
                        )
                    except TypeError:
                        ql.update(
                            state=cell.cell_id,
                            action=choice,
                            reward=float(r),
                            next_state=cell.cell_id,
                            next_actions=actions,
                            done=True,
                        )
                
                # Calculate episode statistics
                episode_data["avg_latency"] = sum(episode_latencies) / len(episode_latencies) if episode_latencies else 0.0
                episode_data["avg_energy"] = sum(episode_energies) / len(episode_energies) if episode_energies else 0.0
                episode_data["success_rate"] = episode_data["successful_actions"] / training_steps
                episode_data["edge_action_ratio"] = episode_data["edge_actions"] / training_steps
                episode_data["cloud_action_ratio"] = episode_data["cloud_actions"] / training_steps
                episode_data["local_action_ratio"] = episode_data["local_actions"] / training_steps
                
                # Add to training backlog
                training_backlog.append(episode_data)
                
                print(f"Completed DQN training episode {training_episodes}:")
                print(f"  - Edge actions: {episode_data['edge_actions']} ({episode_data['edge_action_ratio']:.1%})")
                print(f"  - Cloud actions: {episode_data['cloud_actions']} ({episode_data['cloud_action_ratio']:.1%})")
                print(f"  - Local actions: {episode_data['local_actions']} ({episode_data['local_action_ratio']:.1%})")
                print(f"  - Success rate: {episode_data['success_rate']:.1%}")
                print(f"  - Avg latency: {episode_data['avg_latency']:.3f}s")
                print(f"  - Avg energy: {episode_data['avg_energy']:.3f}J")
                print(f"  - Total reward: {episode_data['total_reward']:.3f}")
                print(f"  - Epsilon: {episode_data['training_steps'][-1]['epsilon']:.3f}")

            iteration += 1

    # Duration (fixed if not provided)
    if sim_time_s is None:
        sim_time_s = 3600.0

    # Run
    env.process(driver())
    env.run(until=sim_time_s)

    # Full report with plots
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.clf()
        text = [
            "Experiment 03 - Edge-Focused MEC with DQN Training Episodes",
            f"Sim time: {sim_time_s}s",
            f"Edge offload bias: {edge_offload_bias}",
            f"Cloud penalty: {cloud_penalty}",
            f"Local penalty: {local_penalty}",
            f"Training episodes: {len(training_backlog)}",
            "",
            "Summary:",
        ]
        sumr = metrics.summary()
        for k, v in sumr.items():
            text.append(f"- {k}: {v}")
        plt.axis('off')
        plt.text(0.05, 0.95, "\n".join(text), va='top', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Timeseries plots
        if metrics.steps:
            times = [s.time_s for s in metrics.steps]
            avg_latency = [s.avg_latency_s or 0.0 for s in metrics.steps]
            throughput = [s.throughput_tasks_per_s for s in metrics.steps]
            off_ratio = [s.offloaded_ratio for s in metrics.steps]
            cloud_ratio = [s.cloud_offload_ratio for s in metrics.steps]
            energy = [s.total_energy_j for s in metrics.steps]
            sched_time = [s.sched_time_s for s in metrics.steps]
            sched_per_decision = [s.sched_time_per_decision_s for s in metrics.steps]
            accept = [(s.acceptance_ratio if s.acceptance_ratio is not None else 0.0) for s in metrics.steps]
            failure = [(s.failure_ratio if s.failure_ratio is not None else 0.0) for s in metrics.steps]
            lb_cv = [(s.load_balance_cv if s.load_balance_cv is not None else 0.0) for s in metrics.steps]
            energy_eff = [((s.completed_tasks / max(1.0, (energy[idx] - (energy[idx-1] if idx>0 else 0.0)))) if (idx < len(energy)) else 0.0) for idx, s in enumerate(metrics.steps)]

            plot_latency(times, avg_latency)
            pdf.savefig(plt.gcf()); plt.close()
            plot_throughput(times, throughput)
            pdf.savefig(plt.gcf()); plt.close()
            plot_offloading(times, off_ratio, cloud_ratio)
            pdf.savefig(plt.gcf()); plt.close()
            plot_energy(times, energy)
            pdf.savefig(plt.gcf()); plt.close()
            plot_scheduling_time(times, sched_time, sched_per_decision)
            pdf.savefig(plt.gcf()); plt.close()
            plot_accept_failure(times, accept, failure)
            pdf.savefig(plt.gcf()); plt.close()
            plot_load_balance(times, lb_cv)
            pdf.savefig(plt.gcf()); plt.close()
            plot_energy_efficiency(times, energy_eff)
            pdf.savefig(plt.gcf()); plt.close()

        # Distribution plots
        lat_all = [t.latency_s for t in metrics.tasks.values() if t.latency_s is not None]
        if lat_all:
            plot_latency_hist(lat_all); pdf.savefig(plt.gcf()); plt.close()
            plot_latency_cdf(lat_all); pdf.savefig(plt.gcf()); plt.close()

        # Optional JSON/CSV outputs
        if summary_json_path is not None:
            try:
                import json
                summary_data = metrics.summary()
                summary_data["training_backlog"] = training_backlog
                summary_data["training_episodes_completed"] = len(training_backlog)
                with open(summary_json_path, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)
            except Exception:
                pass
        if timeseries_csv_path is not None and metrics.steps:
            try:
                import csv
                with open(timeseries_csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "time_s","avg_latency_s","throughput_tps","offloaded_ratio","cloud_offload_ratio",
                        "total_energy_j","sched_time_s","sched_time_per_decision_s","decisions","arrivals",
                        "acceptance_ratio","failure_ratio","busy_time_total_s","load_balance_cv"
                    ])
                    for s in metrics.steps:
                        w.writerow([
                            s.time_s,
                            s.avg_latency_s if s.avg_latency_s is not None else 0.0,
                            s.throughput_tasks_per_s,
                            s.offloaded_ratio,
                            s.cloud_offload_ratio,
                            s.total_energy_j,
                            s.sched_time_s,
                            s.sched_time_per_decision_s,
                            s.decisions,
                            s.arrivals,
                            s.acceptance_ratio if s.acceptance_ratio is not None else 0.0,
                            s.failure_ratio if s.failure_ratio is not None else 0.0,
                            s.busy_time_total_s,
                            s.load_balance_cv if s.load_balance_cv is not None else 0.0,
                        ])
            except Exception:
                pass

    return pdf_path


if __name__ == "__main__":
    out = run_experiment_03(sim_time_s=3600.0, edges_count=40, cells_per_edge=60)
    print(f"Saved report to: {out}")






