from __future__ import annotations

from typing import List, Tuple
import numpy as np

from game.enums import Cell, Noise, BOARD_SIZE

# NOISE_PROBS[cell_type] = (P(squeak), P(scratch), P(squeal))
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

DIST_OFFSETS = (-1, 0, 1, 2)
DIST_PROBS = (0.12, 0.70, 0.12, 0.06)
HEADSTART_MOVES = 1000


def _pos_to_idx(pos: Tuple[int, int]) -> int:
    return pos[1] * BOARD_SIZE + pos[0]


def _idx_to_pos(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RatTracker:
    """
    Belief tracker for the hidden rat.

    Key fixes over the original version:
    - Includes BLOCKED cells, because the rat can live under them.
    - Resets to the true 1000-step spawn distribution, not uniform.
    - Handles the "distance never less than zero" clamp exactly.
    - Uses numpy for much faster belief updates.
    """

    def __init__(self, transition_matrix, board):
        self.N = BOARD_SIZE * BOARD_SIZE
        self.T = np.asarray(transition_matrix, dtype=np.float64)
        self._confirmed_empty = np.zeros(self.N, dtype=bool)

        # Precompute true spawn distribution: start at (0,0), then move 1000 times.
        spawn = np.zeros(self.N, dtype=np.float64)
        spawn[_pos_to_idx((0, 0))] = 1.0
        belief = spawn
        for _ in range(HEADSTART_MOVES):
            belief = belief @ self.T
        self._spawn_belief = belief / belief.sum()
        self.belief = self._spawn_belief.copy()

        # Precompute cell-type indices for quick observation updates.
        self._blocked_idxs = []
        self._space_idxs = []
        for i in range(self.N):
            pos = _idx_to_pos(i)
            if board.get_cell(pos) == Cell.BLOCKED:
                self._blocked_idxs.append(i)
            else:
                self._space_idxs.append(i)

    def update(self, sensor_data: Tuple, worker_pos: Tuple[int, int], board):
        noise, observed_distance = sensor_data

        self._predict()
        self._observe_noise(noise, board)
        self._observe_distance(observed_distance, worker_pos)

        self.belief[self._confirmed_empty] = 0.0
        self._normalize(reset_to_spawn=True)

    def record_our_miss(self, search_loc: Tuple[int, int]):
        idx = _pos_to_idx(search_loc)
        self._confirmed_empty[idx] = True
        self.belief[idx] = 0.0
        self._normalize(reset_to_spawn=False)

    def record_opponent_miss(self, search_loc: Tuple[int, int]):
        idx = _pos_to_idx(search_loc)
        self._confirmed_empty[idx] = True
        self.belief[idx] = 0.0
        self._normalize(reset_to_spawn=False)

    def record_hit(self, board=None):
        self._confirmed_empty[:] = False
        self.belief = self._spawn_belief.copy()

    def should_search(self, turns_left: int, best_board_value: float = 0.0) -> bool:
        ev = self.best_search_ev()
        # Require search EV to beat at least some of the likely board value.
        margin = 0.35 * max(best_board_value, 0.0)

        if turns_left <= 5:
            return ev >= max(1.0, margin)
        if turns_left <= 12:
            return ev >= max(0.75, margin)
        return ev >= max(0.4, margin)

    def best_search_target(self) -> Tuple[int, int]:
        idx = int(np.argmax(self.belief))
        return _idx_to_pos(idx)

    def best_search_ev(self) -> float:
        p = float(np.max(self.belief))
        return 4.0 * p - 2.0 * (1.0 - p)

    def top_k_targets(self, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        k = max(1, min(k, self.N))
        idxs = np.argpartition(-self.belief, k - 1)[:k]
        idxs = idxs[np.argsort(-self.belief[idxs])]
        return [(_idx_to_pos(int(i)), float(self.belief[i])) for i in idxs]

    def get_belief(self) -> List[float]:
        return self.belief.tolist()

    def get_belief_at(self, pos: Tuple[int, int]) -> float:
        return float(self.belief[_pos_to_idx(pos)])

    def _predict(self):
        self.belief = self.belief @ self.T

    def _observe_noise(self, noise: Noise, board):
        noise_idx = int(noise)
        weights = np.empty(self.N, dtype=np.float64)
        for i in range(self.N):
            pos = _idx_to_pos(i)
            cell_type = board.get_cell(pos)
            weights[i] = NOISE_PROBS[cell_type][noise_idx]
        self.belief *= weights

    def _observe_distance(self, observed: int, worker_pos: Tuple[int, int]):
        weights = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            rat_pos = _idx_to_pos(i)
            actual = _manhattan(worker_pos, rat_pos)
            weights[i] = self._distance_likelihood(observed, actual)
        self.belief *= weights

    def _distance_likelihood(self, observed: int, actual: int) -> float:
        prob = 0.0
        for off, p in zip(DIST_OFFSETS, DIST_PROBS):
            reported = actual + off
            if reported < 0:
                reported = 0
            if reported == observed:
                prob += p
        return prob

    def _normalize(self, reset_to_spawn: bool):
        total = float(self.belief.sum())
        if total > 1e-15:
            self.belief /= total
            return
        self.belief = self._spawn_belief.copy() if reset_to_spawn else self.belief
