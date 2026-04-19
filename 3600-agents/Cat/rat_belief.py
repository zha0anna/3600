from __future__ import annotations

from typing import Tuple

import numpy as np

from game.enums import BOARD_SIZE, Cell
from game.rat import DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS, NOISE_PROBS


class RatBeliefHMM:
	def __init__(self, transition_matrix):
		self.num_states = BOARD_SIZE * BOARD_SIZE
		self.transition = np.asarray(transition_matrix, dtype=np.float64)

		if self.transition.shape != (self.num_states, self.num_states):
			raise ValueError(
				f"Expected transition matrix shape {(self.num_states, self.num_states)}, "
				f"got {self.transition.shape}"
			)

		self.belief = np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)
		self._distance_prob = {
			offset: prob
			for offset, prob in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS)
		}

	def index_to_pos(self, index: int) -> Tuple[int, int]:
		return index % BOARD_SIZE, index // BOARD_SIZE

	def pos_to_index(self, pos: Tuple[int, int]) -> int:
		return pos[1] * BOARD_SIZE + pos[0]

	def _normalize(self, arr: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
		arr = np.clip(arr, 0.0, None)
		total = arr.sum()
		if total > 0.0 and np.isfinite(total):
			return arr / total

		if fallback is not None:
			fallback = np.clip(fallback, 0.0, None)
			fallback_total = fallback.sum()
			if fallback_total > 0.0 and np.isfinite(fallback_total):
				return fallback / fallback_total

		return np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)

	def _noise_likelihood(self, board, noise_value: int) -> np.ndarray:
		probs = np.empty(self.num_states, dtype=np.float64)
		for idx in range(self.num_states):
			pos = self.index_to_pos(idx)
			cell = board.get_cell(pos)
			cell_probs = NOISE_PROBS.get(cell, NOISE_PROBS[Cell.SPACE])
			probs[idx] = cell_probs[noise_value]
		return probs

	def _distance_likelihood(self, board, observed_distance: int) -> np.ndarray:
		px, py = board.player_worker.get_location()
		probs = np.empty(self.num_states, dtype=np.float64)
		for idx in range(self.num_states):
			rx, ry = self.index_to_pos(idx)
			true_distance = abs(px - rx) + abs(py - ry)
			offset = observed_distance - true_distance
			probs[idx] = self._distance_prob.get(offset, 0.0)
		return probs

	def _observation_likelihood(self, board, sensor_data: Tuple) -> np.ndarray:
		noise, observed_distance = sensor_data
		noise_value = int(noise)
		noise_like = self._noise_likelihood(board, noise_value)
		dist_like = self._distance_likelihood(board, int(observed_distance))
		return noise_like * dist_like

	def apply_search_evidence(self, search_loc: Tuple[int, int] | None, search_result: bool | None) -> None:
		if search_loc is None or search_result is None:
			return

		if search_result:
			# Rat respawns from (0, 0) and moves before the next sensor sample.
			self.reset_to_spawn()
			return

		idx = self.pos_to_index(search_loc)
		updated = self.belief.copy()
		updated[idx] = 0.0
		self.belief = self._normalize(updated)

	def reset_to_spawn(self) -> None:
		self.belief = np.zeros(self.num_states, dtype=np.float64)
		self.belief[self.pos_to_index((0, 0))] = 1.0

	def update(self, board, sensor_data: Tuple) -> np.ndarray:
		predicted = np.matmul(self.belief, self.transition)
		predicted = self._normalize(predicted)

		likelihood = self._observation_likelihood(board, sensor_data)
		posterior = predicted * likelihood
		self.belief = self._normalize(posterior, fallback=predicted)
		return self.belief

	def estimate(self) -> Tuple[Tuple[int, int], float, float]:
		best_index = int(np.argmax(self.belief))
		confidence = float(self.belief[best_index])

		safe_belief = np.clip(self.belief, 1e-12, 1.0)
		entropy = float(-np.sum(self.belief * np.log(safe_belief)))

		return self.index_to_pos(best_index), confidence, entropy

	def top_k(self, k: int = 5):
		k = max(1, min(k, self.num_states))
		top_indices = np.argpartition(self.belief, -k)[-k:]
		top_indices = top_indices[np.argsort(self.belief[top_indices])[::-1]]
		return [(self.index_to_pos(int(i)), float(self.belief[i])) for i in top_indices]
