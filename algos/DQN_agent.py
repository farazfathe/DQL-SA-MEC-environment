from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Hashable, List, Sequence, Tuple
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


State = Hashable
Action = Hashable


def default_featurizer(state: State, action: Action) -> torch.Tensor:
	STATE_DIM = 128
	ACTION_DIM = 256
	sh = abs(hash(state)) % STATE_DIM
	ah = abs(hash(str(action))) % ACTION_DIM
	feat = torch.zeros(STATE_DIM + ACTION_DIM, dtype=torch.float32)
	feat[sh] = 1.0
	feat[STATE_DIM + ah] = 1.0
	return feat


class MLPQ(nn.Module):
	def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Accept shape (D,) or (N, D); return scalar or (N,)
		is_single = (x.dim() == 1)
		if is_single:
			x = x.unsqueeze(0)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		out = self.fc3(x)  # (N, 1)
		out = out[:, 0]    # (N,)
		return out[0] if is_single else out


@dataclass
class DQNAgent:
	alpha: float = 5e-4
	gamma: float = 0.99
	epsilon: float = 1.0
	epsilon_end: float = 0.1
	epsilon_decay_steps: int = 100000
	input_dim: int = 384
	hidden_dim: int = 256
	featurizer: Callable[[State, Action], torch.Tensor] = default_featurizer
	buffer_capacity: int = 100000
	batch_size: int = 64
	learn_start: int = 1000
	update_freq: int = 1
	target_update_freq: int = 1000
	device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
	# Options
	use_double_dqn: bool = True
	use_per: bool = True
	per_alpha: float = 0.6
	per_beta: float = 0.4
	per_beta_increment: float = 1e-6
	per_eps: float = 1e-3

	# runtime
	steps: int = 0
	replay: Deque[Tuple[State, Action, float, State, Tuple[Action, ...], bool]] = field(default_factory=lambda: deque(maxlen=100000))
	priorities: Deque[float] = field(default_factory=lambda: deque(maxlen=100000))

	def __post_init__(self):
		self.q = MLPQ(self.input_dim, self.hidden_dim).to(self.device)
		self.tq = MLPQ(self.input_dim, self.hidden_dim).to(self.device)
		self.tq.load_state_dict(self.q.state_dict())
		self.opt = torch.optim.Adam(self.q.parameters(), lr=self.alpha)

	def get_q(self, state: State, action: Action) -> float:
		with torch.no_grad():
			x = self.featurizer(state, action).to(self.device)
			q = self.q(x)
			return float(q.item())

	def best_action(self, state: State, actions: Sequence[Action]) -> Tuple[Action, float]:
		if not actions:
			raise ValueError("actions must be non-empty")
		xs = torch.stack([self.featurizer(state, a) for a in actions], dim=0).to(self.device)
		with torch.no_grad():
			qs = self.q(xs)
		max_idx = int(torch.argmax(qs).item())
		return actions[max_idx], float(qs[max_idx].item())

	def select_action(self, state: State, actions: Sequence[Action]) -> Action:
		if random.random() < self.epsilon:
			return random.choice(list(actions))
		best_a, _ = self.best_action(state, actions)
		return best_a

	def _sync_target(self):
		self.tq.load_state_dict(self.q.state_dict())

	def update(self, state: State, action: Action, reward: float, next_state: State, next_actions: Sequence[Action], done: bool | None = None) -> None:
		# Step, epsilon schedule (linear per step)
		self.steps += 1
		frac = max(0.0, (self.epsilon_decay_steps - self.steps) / float(max(1, self.epsilon_decay_steps)))
		self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * frac

		# Append transition
		if done is None:
			done = (len(next_actions) == 0)
		self.replay.append((state, action, reward, next_state, tuple(next_actions), bool(done)))
		# Initial priority: max existing or 1
		if self.use_per:
			p0 = max(self.priorities) if len(self.priorities) > 0 else 1.0
			self.priorities.append(p0)
		else:
			self.priorities.append(1.0)

		# Target network sync
		if self.steps % self.target_update_freq == 0:
			self._sync_target()

		# Not enough to learn yet
		if self.steps < self.learn_start:
			return
		if (self.steps % self.update_freq) != 0:
			return
		if len(self.replay) < self.batch_size:
			return

		# Sample minibatch (PER or uniform)
		if self.use_per and len(self.priorities) >= self.batch_size:
			pr = np.array(self.priorities, dtype=np.float64)
			probs = (pr + self.per_eps) ** self.per_alpha
			probs /= probs.sum()
			idxs = np.random.choice(len(self.replay), size=self.batch_size, replace=False, p=probs)
			batch = [self.replay[i] for i in idxs]
			# Importance-sampling weights
			self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
			weights = (len(self.replay) * probs[idxs]) ** (-self.per_beta)
			weights = weights / (weights.max() + 1e-8)
			w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
		else:
			batch = random.sample(self.replay, self.batch_size)
			w_t = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
		# Build tensors for current (state, action)
		x_batch = torch.stack([self.featurizer(s, a) for (s, a, *_rest) in batch], dim=0).to(self.device)
		r_batch = torch.tensor([r for (_s, _a, r, *_rest) in batch], dtype=torch.float32, device=self.device)
		d_batch = torch.tensor([1.0 if d else 0.0 for (*_sa, d) in batch], dtype=torch.float32, device=self.device)

		# Vectorized next-state Q: Double DQN (online selects, target evaluates) or standard target max
		flat_x2: List[torch.Tensor] = []
		seg_sizes: List[int] = []
		for (_s, _a, _r, s2, a2_list, _d) in batch:
			seg_sizes.append(len(a2_list))
			for a2 in a2_list:
				flat_x2.append(self.featurizer(s2, a2))
		if flat_x2:
			x2_all = torch.stack(flat_x2, dim=0).to(self.device)
			with torch.no_grad():
				if self.use_double_dqn:
					# Online picks argmax indices per segment
					q2_online = self.q(x2_all)
					q2_target = self.tq(x2_all)
					max_next_vals: List[float] = []
					idx = 0
					for sz in seg_sizes:
						if sz == 0:
							max_next_vals.append(0.0)
							continue
						seg_online = q2_online[idx:idx+sz]
						seg_target = q2_target[idx:idx+sz]
						amax = int(torch.argmax(seg_online).item())
						max_next_vals.append(float(seg_target[amax].item()))
						idx += sz
				else:
					q2_all = self.tq(x2_all)
					max_next_vals = []
					idx = 0
					for sz in seg_sizes:
						if sz == 0:
							max_next_vals.append(0.0)
							continue
						seg = q2_all[idx:idx+sz]
						max_next_vals.append(float(torch.max(seg).item()))
						idx += sz
			max_next_t = torch.tensor(max_next_vals, dtype=torch.float32, device=self.device)
		else:
			max_next_t = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
		# Zero-out next for terminals
		max_next_t = (1.0 - d_batch) * max_next_t
		targets = r_batch + self.gamma * max_next_t

		# Preds
		preds = self.q(x_batch)
		# PER weighting
		losses = F.smooth_l1_loss(preds, targets, reduction='none')
		loss = (w_t * losses).mean()
		self.opt.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
		self.opt.step()

		# Update priorities with new TD-errors
		if self.use_per:
			with torch.no_grad():
				td_err = losses.detach().cpu().numpy() + self.per_eps
				# If uniform sampling, idxs undefined; recompute priorities for a random matching set
				if 'idxs' in locals():
					for i, pe in zip(idxs, td_err):
						self.priorities[i] = float(abs(pe))


