from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional
import random

import simpy
import yaml

# Ensure project root is importable when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

# Use non-interactive backend for headless PDF generation
import matplotlib
matplotlib.use("Agg")

from Env import Cell, EdgeServer, Cloud, GreedyLocalFirstScheduler
from Env.scheduler import QLSASchedulerAdapter
from Env.task import Task
from Env.utils import constant_rate_task_generator, energy_per_cycle_from_power
from Env.utils import poisson_task_generator

from algos.DQN_agent import DQNAgent
from algos.simulated_annealing import SimulatedAnnealing
from algos.QL_SA_wrapper import QLSAWrapper

from analysis.Metrics import MetricsLogger
from analysis.plots import (
	plot_latency,
	plot_throughput,
	plot_offloading,
	plot_energy,
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


def run_experiment_02(
	sim_time_s: Optional[float] = None,
	env_cfg_path: str = os.path.join("data", "env.yaml"),
	algo_cfg_path: str = os.path.join("data", "algo.yaml"),
	pdf_path: str = os.path.join("data", "Result_exp02.pdf"),
	edges_count: int = 200,
	cells_per_edge: int = 20,
	lambda_per_s: float = 10.0,
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
		e.start(env)
		edges.append(e)

	cloud = Cloud(
		cpu_rate_cycles_per_s=cloud_cpu_rate,
		energy_joules=cloud_energy_j,
		compute_energy_j_per_cycle=cloud_compute_energy_per_cycle,
	)
	cloud.start(env)

	# Metrics hooks for edge/cloud
	metrics = MetricsLogger()

	def on_edge_done(task: Task, energy_cost: float):
		metrics.mark_completed(task.task_id, float(env.now), energy_cost)

	def on_cloud_done(task: Task, energy_cost: float):
		metrics.mark_completed(task.task_id, float(env.now), energy_cost)

	for e in edges:
		e.on_task_completed = on_edge_done
	cloud.on_task_completed = on_cloud_done

	# Task generation profile
	bit_size_bits = 1_000_000  # 1 Mb per task
	memory_bytes = 64 * 1024 * 1024  # 64 MB
	cpu_cycles = int(5e8)  # 0.5 Gcycles
	deadline_s = 1.0

	# cells per edge => total cells is parameterized
	cells: List[Cell] = []
	for i, e in enumerate(edges):
		# Poisson arrivals per cell (lambda tuned per parameter)
		gen = poisson_task_generator(
			factory=make_task_factory(bit_size_bits, memory_bytes, cpu_cycles, deadline_s),
			lambda_per_s=float(lambda_per_s),
		)
		for j in range(int(cells_per_edge)):
			cell_idx = i * 20 + j
			c = Cell(
				cell_id=f"cell-{cell_idx}",
				cpu_rate_cycles_per_s=cell_cpu_rate,
				battery_energy_joules=cell_battery_j,
				task_generator=gen,
				edge_id=e.edge_id,
			)
			c.compute_energy_j_per_cycle = cell_energy_per_cycle
			c.tx_energy_j_per_bit_to_edge = float(env_conf["cell"]["tx_energy_per_bit"]) 
			cells.append(c)

	# Start cells
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
	scheduler = QLSASchedulerAdapter(policy=policy, edge_ids=[e.edge_id for e in edges], bias_edge=True)

	# --- Driver ---
	def driver():
		window_s = 1.0
		iteration = 0
		last_objective = 0.0
		while True:
			completed_this_tick = 0
			offloaded_count = 0
			cloud_offloaded_count = 0
			latencies: List[float] = []
			waits: List[float] = []
			energy_this_tick = 0.0

			# Process arrivals for each cell (Poisson per 1s bucket)
			for c in cells:
				new_tasks: List[Task] = list(c.task_generator(float(env.now)))
				for t in new_tasks:
					t.source_cell_id = c.cell_id
					# Let the scheduler decide: local, which edge, or cloud
					actions = ["local", "cloud"] + [("edge", e.edge_id) for e in edges]
					choice = policy.select_action(c.cell_id, actions)
					if choice == "local":
						# Try local; if infeasible, attempt offload to default edge
						if c.execute_locally(t, float(env.now)):
							completed_this_tick += 1
							latencies.append(t.finished_at - t.created_at)
							if t.started_at is not None:
								waits.append((t.started_at - (t.queued_at or t.created_at)) if t.queued_at is not None else 0.0)
							energy = t.cpu_cycles * c.compute_energy_j_per_cycle
							energy_this_tick += energy
							metrics.log_task(
								task_id=t.task_id,
								source_cell_id=t.source_cell_id,
								created_at=t.created_at,
								started_at=t.started_at,
								finished_at=t.finished_at,
								deadline=t.deadline,
								was_offloaded=False,
								was_to_cloud=False,
								energy_j=energy,
							)
						else:
							choice = ("edge", c.edge_id) if c.edge_id is not None else ("cloud",)
					if isinstance(choice, tuple) and choice[0] == "edge":
						# Model uplink transmission delay and energy before putting into edge queue
						tx_time = t.bit_size / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0
						cost = c.prepare_offload_to_edge(t)
						if cost is not None:
							def _send_to_edge(task=t, edge_id=choice[1], energy_cost=cost, tx_delay=tx_time):
								# After TX delay, push to edge queue
								yield env.timeout(tx_delay)
								idx = int(edge_id.split("-")[-1])
								edges[idx].put(task)
							metrics.log_task(
								task_id=t.task_id,
								source_cell_id=t.source_cell_id,
								created_at=t.created_at,
								started_at=None,
								finished_at=None,
								deadline=t.deadline,
								was_offloaded=True,
								was_to_cloud=False,
								energy_j=cost,
							)
							offloaded_count += 1
							energy_this_tick += cost
							env.process(_send_to_edge())
					elif choice == "cloud":
						tx_time = t.bit_size / bw_iot_to_mec if bw_iot_to_mec > 0 else 0.0
						cost = c.prepare_offload_to_edge(t)
						if cost is not None:
							def _send_to_cloud(task=t, energy_cost=cost, tx_delay=tx_time):
								yield env.timeout(tx_delay)
								cloud.put(task)
							metrics.log_task(
								task_id=t.task_id,
								source_cell_id=t.source_cell_id,
								created_at=t.created_at,
								started_at=None,
								finished_at=None,
								deadline=t.deadline,
								was_offloaded=True,
								was_to_cloud=True,
								energy_j=cost,
							)
							cloud_offloaded_count += 1
							energy_this_tick += cost
							env.process(_send_to_cloud())

			# Advance time window for metrics aggregation
			yield env.timeout(window_s)

			# Objective proxy (use averaged latencies and offload ratio)
			avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
			total_new = max(1, completed_this_tick + offloaded_count)
			edge_offload_ratio = offloaded_count / total_new
			Z = 0.6 * avg_lat + 0.2 * (1.0 - edge_offload_ratio) + 0.2 * (energy_this_tick)
			reward = -(Z - last_objective)
			next_state = iteration + 1
			ql.update(state=iteration, action="schedule", reward=reward, next_state=next_state, next_actions=["schedule"]) 

			metrics.log_step(
				time_s=float(env.now),
				completed_tasks=completed_this_tick,
				offloaded=offloaded_count,
				cloud_offloaded=cloud_offloaded_count,
				energy_j=energy_this_tick,
				latencies=latencies,
				waits=waits,
				server_utilization={e.edge_id: 0.0 for e in edges},
				queue_lengths={e.edge_id: len(e.queue) for e in edges},
				window_s=window_s,
			)

			iteration += 1

	# Duration
	if sim_time_s is None:
		sim_time_s = float(env_conf["simulation"].get("duration", 3600.0))

	# Run
	env.process(driver())
	env.run(until=sim_time_s)

	# Report
	os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
	with PdfPages(pdf_path) as pdf:
		fig = plt.figure(figsize=(8.27, 11.69))
		fig.clf()
		text = [
			"Experiment 02 - MEC with DQL+SA (env.yaml & algo.yaml)",
			f"Sim time: {sim_time_s}s",
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

		times = [s.time_s for s in metrics.steps]
		avg_latency = [s.avg_latency_s or 0 for s in metrics.steps]
		throughput = [s.throughput_tasks_per_s for s in metrics.steps]
		off_ratio = [s.offloaded_ratio for s in metrics.steps]
		cloud_ratio = [s.cloud_offload_ratio for s in metrics.steps]
		energy = [s.total_energy_j for s in metrics.steps]

		plot_latency(times, avg_latency)
		pdf.savefig(plt.gcf())
		plt.close()

		plot_throughput(times, throughput)
		pdf.savefig(plt.gcf())
		plt.close()

		plot_offloading(times, off_ratio, cloud_ratio)
		pdf.savefig(plt.gcf())
		plt.close()

		plot_energy(times, energy)
		pdf.savefig(plt.gcf())
		plt.close()

	return pdf_path


if __name__ == "__main__":
	# Always use duration from env.yaml
	path = run_experiment_02(sim_time_s=None)
	print(f"Saved report to: {path}")


