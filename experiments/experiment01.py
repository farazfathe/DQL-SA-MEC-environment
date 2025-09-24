from __future__ import annotations

import os
import sys
from typing import List, Tuple

import simpy

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

from algos.Q_learning_agent import QLearningAgent
from algos.simulated_annealing import SimulatedAnnealing
from algos.QL_SA_wrapper import QLSAWrapper

from analysis.Metrics import MetricsLogger
from analysis.plots import (
	plot_latency,
	plot_throughput,
	plot_offloading,
	plot_energy,
	plot_sa_convergence,
)

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


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


def run_experiment(sim_time_s: float = 60.0, pdf_path: str = os.path.join("data", "Result_exp01.pdf")) -> str:
	# --- SimPy env ---
	env = simpy.Environment()

	# --- Infrastructure parameters (example values) ---
	# Make device weaker to encourage offloading
	cell_cpu_rate = 1e9  # 1 GHz equivalent cycles/s
	cell_power_w = 2.0  # watts average when computing (higher cost)
	cell_energy_per_cycle = energy_per_cycle_from_power(cell_power_w, cell_cpu_rate)
	cell_battery_j = 500.0  # larger so many transmissions are possible

	edge_cpu_rate = 5e10  # 50 GHz cycles/s (more capable edge)
	edge_energy_j = 1e6
	edge_compute_energy_per_cycle = 5e-12  # cheaper to compute at edge

	cloud_cpu_rate = 8e10
	cloud_energy_j = 1e9
	cloud_compute_energy_per_cycle = 5e-12

	# --- Nodes ---
	edge = EdgeServer(
		edge_id="edge-0",
		cpu_rate_cycles_per_s=edge_cpu_rate,
		energy_joules=edge_energy_j,
		compute_energy_j_per_cycle=edge_compute_energy_per_cycle,
	)
	edge.start(env)

	cloud = Cloud(
		cpu_rate_cycles_per_s=cloud_cpu_rate,
		energy_joules=cloud_energy_j,
		compute_energy_j_per_cycle=cloud_compute_energy_per_cycle,
	)
	cloud.start(env)

	# Hook completion callbacks for metrics
	def on_edge_done(task: Task, energy_cost: float):
		metrics.mark_completed(task.task_id, float(env.now), energy_cost)

	def on_cloud_done(task: Task, energy_cost: float):
		metrics.mark_completed(task.task_id, float(env.now), energy_cost)

	edge.on_task_completed = on_edge_done
	cloud.on_task_completed = on_cloud_done

	# --- Task generation ---
	bit_size_bits = 1_000_000  # 1 Mb (lighter to transmit)
	memory_bytes = 64 * 1024 * 1024  # 64 MB
	cpu_cycles = int(5e8)  # 0.5 Gcycle (reduced per-task density)
	deadline_s = 1.0
	# Generate ~4200 tasks in 60 seconds
	gen = constant_rate_task_generator(
		factory=make_task_factory(bit_size_bits, memory_bytes, cpu_cycles, deadline_s),
		tasks_per_tick=70,
	)

	cell = Cell(
		cell_id="cell-0",
		cpu_rate_cycles_per_s=cell_cpu_rate,
		battery_energy_joules=cell_battery_j,
		task_generator=gen,
		edge_id=edge.edge_id,
	)
	cell.compute_energy_j_per_cycle = cell_energy_per_cycle
	cell.tx_energy_j_per_bit_to_edge = 1e-9  # make offloading cheaper

	# --- Policy (QL + SA) ---
	ql = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)
	sa = SimulatedAnnealing(initial_temperature=1.0, cooling_rate=0.97, min_temperature=1e-3, steps_per_call=25)
	policy = QLSAWrapper(ql=ql, sa=sa)
	scheduler = QLSASchedulerAdapter(policy=policy, edge_ids=[edge.edge_id], bias_edge=True)

	# QL over SA cooling schedule (actions) and simple state
	actions_sa = ["cool_fast", "hold", "cool_slow"]

	def choose_sa_action(iteration: int, last_improvement: float) -> str:
		# State as a tuple reduced to discrete bins
		state = (iteration // 5, 1 if last_improvement < 0 else (0 if last_improvement == 0 else -1))
		return ql.select_action(state, actions_sa)

	def apply_sa_action(action: str) -> None:
		# Adjust cooling_rate within bounds to steer exploration/exploitation
		if action == "cool_fast":
			sa.cooling_rate = max(0.90, sa.cooling_rate * 0.97)
		elif action == "cool_slow":
			sa.cooling_rate = min(0.999, sa.cooling_rate * 1.01)
		else:
			pass

	# --- Metrics ---
	metrics = MetricsLogger()

	# --- Driver process ---
	def driver():
		window_s = 1.0
		iteration = 0
		last_objective = 0.0
		last_action = "hold"
		while env.now < sim_time_s:
			# Generate tasks for this tick
			now = float(env.now)
			new_tasks: List[Task] = list(cell.task_generator(now))
			for t in new_tasks:
				t.source_cell_id = cell.cell_id

			# Decide placement using scheduler
			local, to_edge, to_cloud = scheduler.place(cell.cell_id, new_tasks)

			completed_this_tick = 0
			offloaded_count = 0
			cloud_offloaded_count = 0
			latencies: List[float] = []
			waits: List[float] = []
			energy_this_tick = 0.0

			# --- RL: decide SA cooling adaptation before acting this step ---
			improvement_proxy = 0.0  # will be updated after computing objective
			sa_action = choose_sa_action(iteration, improvement_proxy)
			apply_sa_action(sa_action)

			# Handle local executions serially (one-by-one)
			for task in local:
				# Check if can complete before deadline
				exec_time = task.estimated_compute_time(cell.cpu_rate_cycles_per_s)
				finish_time = env.now + exec_time
				energy_cost = task.cpu_cycles * cell.compute_energy_j_per_cycle
				if energy_cost <= cell.battery_energy_joules and finish_time <= task.deadline:
					task.mark_started(env.now)
					yield env.timeout(exec_time)
					task.mark_completed(env.now)
					cell.battery_energy_joules -= energy_cost
					completed_this_tick += 1
					energy_this_tick += energy_cost
					latencies.append(task.finished_at - task.created_at)
					waits.append(task.started_at - task.created_at)
					metrics.log_task(
						task_id=task.task_id,
						source_cell_id=task.source_cell_id,
						created_at=task.created_at,
						started_at=task.started_at,
						finished_at=task.finished_at,
						deadline=task.deadline,
						was_offloaded=False,
						was_to_cloud=False,
						energy_j=energy_cost,
					)
				else:
					# Try to offload to edge if local infeasible
					cost = cell.prepare_offload_to_edge(task)
					if cost is not None:
						edge.put(task)
						offloaded_count += 1
						energy_this_tick += cost
						metrics.log_task(
							task_id=task.task_id,
							source_cell_id=task.source_cell_id,
							created_at=task.created_at,
							started_at=None,
							finished_at=None,
							deadline=task.deadline,
							was_offloaded=True,
							was_to_cloud=False,
							energy_j=cost,
						)

			# Handle explicit edge offloads
			for edge_id, task in to_edge:
				cost = cell.prepare_offload_to_edge(task)
				if cost is not None:
					edge.put(task)
					offloaded_count += 1
					energy_this_tick += cost
					metrics.log_task(
						task_id=task.task_id,
						source_cell_id=task.source_cell_id,
						created_at=task.created_at,
						started_at=None,
						finished_at=None,
						deadline=task.deadline,
						was_offloaded=True,
						was_to_cloud=False,
						energy_j=cost,
					)

			# Handle cloud offloads (direct)
			for task in to_cloud:
				cost = cell.prepare_offload_to_edge(task)  # reuse same TX cost model
				if cost is not None:
					cloud.put(task)
					cloud_offloaded_count += 1
					energy_this_tick += cost
					metrics.log_task(
						task_id=task.task_id,
						source_cell_id=task.source_cell_id,
						created_at=task.created_at,
						started_at=None,
						finished_at=None,
						deadline=task.deadline,
						was_offloaded=True,
						was_to_cloud=True,
						energy_j=cost,
					)

			# Advance window and sample completions that finished within window via polling queues
			# Note: Edge/Cloud complete tasks internally; for this simple logger, we cannot directly capture finish times
			# unless we subscribe to events. Here we only log local completions and offloads per window.
			yield env.timeout(max(0.0, window_s - (env.now - now)))

			# --- Objective and RL update (Feedback & Learning) ---
			# Objective Z: lower is better. Encourage edge offloading when possible.
			avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
			queue_penalty = float(len(edge.queue))
			total_new = max(1, len(new_tasks))
			edge_offload_ratio = offloaded_count / total_new
			Z = 0.4 * avg_lat + 0.2 * (energy_this_tick) + 0.2 * queue_penalty + 0.2 * (1.0 - edge_offload_ratio)
			improvement = Z - last_objective
			reward = -improvement  # negative of change (improvement -> negative change -> positive reward)
			# Next state approximation
			next_state = ((iteration + 1) // 5, 1 if improvement < 0 else (0 if improvement == 0 else -1))
			ql.update(
				state=(iteration // 5, 1 if 0.0 < 0 else 0),
				action=sa_action,
				reward=reward,
				next_state=next_state,
				next_actions=actions_sa,
			)
			last_objective = Z
			iteration += 1

			# RL logging
			metrics.log_rl(
				iteration=iteration,
				temperature=sa.initial_temperature,
				objective=Z,
				accepted_worse=None,
				total_moves=None,
				epsilon=ql.epsilon,
				reward=reward,
			)

			# Step-level metrics snapshot
			metrics.log_step(
				time_s=float(env.now),
				completed_tasks=completed_this_tick,
				offloaded=offloaded_count,
				cloud_offloaded=cloud_offloaded_count,
				energy_j=energy_this_tick,
				latencies=latencies,
				waits=waits,
				server_utilization={edge.edge_id: 0.0},
				queue_lengths={edge.edge_id: len(edge.queue)},
				window_s=window_s,
			)

	env.process(driver())
	env.run(until=sim_time_s)

	# --- Reporting ---
	os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
	with PdfPages(pdf_path) as pdf:
		# Summary page
		fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
		fig.clf()
		text = ["Experiment 01 - MEC with QL+SA",
			f"Sim time: {sim_time_s}s",
			"",
			"Summary:"]
		sumr = metrics.summary()
		for k, v in sumr.items():
			text.append(f"- {k}: {v}")
		plt.axis('off')
		plt.text(0.05, 0.95, "\n".join(text), va='top', fontsize=10)
		pdf.savefig(fig)
		plt.close(fig)

		# Time series plots
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

		# SA convergence placeholder (no per-iter logging integrated in this simple run)
		# pdf remains valid even without this plot

	return pdf_path


if __name__ == "__main__":
	path = run_experiment(sim_time_s=300.0)
	print(f"Saved report to: {path}")


