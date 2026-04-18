"""
rat_tracker.py

Hidden Markov Model for tracking the rat.

State: 64-element belief vector where belief[i] = P(rat is at cell i)
       index i = y * 8 + x for position (x, y)

Each turn the HMM does three steps:
  1. PREDICT   — advance belief one step through transition matrix T
  2. OBSERVE   — weight each cell by P(noise | cell type) * P(distance | cell)
  3. NORMALIZE — rescale so belief sums to 1

Special cases:
  - After a HIT (rat caught), belief resets to uniform (new rat spawns)
  - After a MISS (our search or opponent's), that cell is zeroed out
  - Opponent search results are also used to update belief
"""

from typing import Tuple, List
from game.enums import Noise, Cell, BOARD_SIZE

# -----------------------------------------------------------------------
# EXACT PROBABILITIES FROM SPEC
# -----------------------------------------------------------------------

# NOISE_PROBS[cell_type] = (P(squeak), P(scratch), P(squeal))
NOISE_PROBS = {
    Cell.BLOCKED: (0.5,  0.3,  0.2 ),
    Cell.SPACE:   (0.7,  0.15, 0.15),
    Cell.PRIMED:  (0.1,  0.8,  0.1 ),
    Cell.CARPET:  (0.1,  0.1,  0.8 ),
}

# Distance error: observed = actual + offset
# offset:       -1     0     +1    +2
DIST_OFFSETS = (-1,    0,    1,    2  )
DIST_PROBS   = (0.12,  0.70, 0.12, 0.06)


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _pos_to_idx(pos: Tuple[int, int]) -> int:
    return pos[1] * BOARD_SIZE + pos[0]

def _idx_to_pos(idx: int) -> Tuple[int, int]:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _p_distance(observed: int, actual: int) -> float:
    """P(observed distance | actual distance) using the error table."""
    offset = observed - actual
    for off, prob in zip(DIST_OFFSETS, DIST_PROBS):
        if off == offset:
            return prob
    return 0.01  # offset outside the table = impossible


# -----------------------------------------------------------------------
# RAT TRACKER
# -----------------------------------------------------------------------

class RatTracker:

    def __init__(self, transition_matrix, board):
        """
        transition_matrix : 64x64 where T[i][j] = P(rat moves from i to j)
        board             : initial Board object to identify blocked cells
        """
        self.T = transition_matrix
        self.N = BOARD_SIZE * BOARD_SIZE

        # Track which cells are non-blocked for uniform reset
        self._valid_cells = []
        for i in range(self.N):
            pos = _idx_to_pos(i)
            if board.get_cell(pos) != Cell.BLOCKED:
                self._valid_cells.append(i)

        # Start with uniform belief over all non-blocked cells
        self.belief = self._uniform_belief()

        # Cells confirmed empty by our misses or opponent misses
        self._confirmed_empty: set = set()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def update(self, sensor_data: Tuple, worker_pos: Tuple[int, int], board):
        """
        Call once per turn BEFORE deciding what move to make.

        sensor_data : (Noise enum, int estimated distance)
        worker_pos  : our worker's (x, y) position this turn
        board       : current board (needed for cell type lookups)
        """
        noise, distance = sensor_data

        # 1. Predict: advance belief through transition matrix
        self._predict()

        # 2. Observe: weight by noise and distance likelihood
        self._observe_noise(noise, board)
        self._observe_distance(distance, worker_pos)

        # 3. Zero out all confirmed empty cells
        for idx in self._confirmed_empty:
            self.belief[idx] = 0.0

        # 4. Normalize
        self._normalize()

    def record_our_miss(self, search_loc: Tuple[int, int]):
        """
        Call when OUR search came back empty.
        Zeros out that cell permanently until a new rat spawns.
        """
        idx = _pos_to_idx(search_loc)
        self._confirmed_empty.add(idx)
        self.belief[idx] = 0.0
        self._normalize()

    def record_opponent_miss(self, search_loc: Tuple[int, int]):
        """
        Call when the OPPONENT searched and missed.
        We are told their guess and result — use it to update our belief.
        """
        idx = _pos_to_idx(search_loc)
        self._confirmed_empty.add(idx)
        self.belief[idx] = 0.0
        self._normalize()

    def record_hit(self, board):
        """
        Call when the rat was caught (by us or opponent).
        A new rat spawns and runs 1000 steps — reset to uniform belief.
        """
        self._confirmed_empty.clear()
        self.belief = self._uniform_belief()

    def should_search(self, turns_left: int) -> bool:
        """
        Returns True if searching is worth it given current belief and turns left.
        EV of search = p_best * 4 - (1 - p_best) * 2
        EV > 0 when p_best > 1/3.
        We add a margin that increases as turns run out.
        """
        p = max(self.belief)
        ev = p * 4 - (1 - p) * 2

        # Late game: require higher confidence to avoid costly misses
        if turns_left <= 5:
            return ev > 1.5    # p > ~0.67
        elif turns_left <= 10:
            return ev > 1.0    # p > ~0.58
        elif turns_left <= 20:
            return ev > 0.5    # p > ~0.53
        else:
            return ev > 0.2    # p > ~0.50

    def best_search_target(self) -> Tuple[int, int]:
        """Return the cell with the highest probability — best place to search."""
        best_idx = max(range(self.N), key=lambda i: self.belief[i])
        return _idx_to_pos(best_idx)

    def best_search_ev(self) -> float:
        """Expected value of searching the best cell."""
        p = max(self.belief)
        return p * 4 - (1 - p) * 2

    def get_belief(self) -> List[float]:
        return self.belief

    def get_belief_at(self, pos: Tuple[int, int]) -> float:
        return self.belief[_pos_to_idx(pos)]

    # ------------------------------------------------------------------
    # INTERNAL STEPS
    # ------------------------------------------------------------------

    def _predict(self):
        """
        Advance belief one step: new_belief[j] = sum_i belief[i] * T[i][j]
        """
        new_belief = [0.0] * self.N
        for i in range(self.N):
            if self.belief[i] == 0.0:
                continue
            for j in range(self.N):
                new_belief[j] += self.belief[i] * self.T[i][j]
        self.belief = new_belief

    def _observe_noise(self, noise: Noise, board):
        """
        Weight each cell by P(observed noise | rat on that cell type).
        Squeak  → likely SPACE   (p=0.70)
        Scratch → likely PRIMED  (p=0.80)
        Squeal  → likely CARPET  (p=0.80)
        """
        noise_idx = int(noise)  # 0=squeak, 1=scratch, 2=squeal
        for i in range(self.N):
            if self.belief[i] == 0.0:
                continue
            pos = _idx_to_pos(i)
            cell_type = board.get_cell(pos)
            probs = NOISE_PROBS.get(cell_type, NOISE_PROBS[Cell.SPACE])
            self.belief[i] *= probs[noise_idx]

    def _observe_distance(self, observed: int, worker_pos: Tuple[int, int]):
        """
        Weight each cell by P(observed distance | rat at that cell).
        Uses the error table: observed = actual + offset with known probabilities.
        """
        for i in range(self.N):
            if self.belief[i] == 0.0:
                continue
            rat_pos = _idx_to_pos(i)
            actual = _manhattan(worker_pos, rat_pos)
            self.belief[i] *= _p_distance(observed, actual)

    def _normalize(self):
        total = sum(self.belief)
        if total > 1e-10:
            for i in range(self.N):
                self.belief[i] /= total
        else:
            # Belief collapsed — reset to uniform as safety fallback
            self.belief = self._uniform_belief()

    def _uniform_belief(self) -> List[float]:
        """Uniform distribution over all non-blocked cells."""
        belief = [0.0] * self.N
        p = 1.0 / len(self._valid_cells)
        for i in self._valid_cells:
            belief[i] = p
        return belief