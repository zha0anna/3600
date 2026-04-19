from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import random

import numpy as np
from numba import njit, uint64

from game.enums import BOARD_SIZE, Cell, Direction, MoveType, Result
from game.move import Move
from .rat_belief import RatBeliefHMM

# ==========================================
# NUMBA JIT OPTIMIZED HEURISTICS (BITBOARDS)
# ==========================================

@njit(cache=True)
def fast_dist_to_center(px: int, py: int) -> int:
    c1 = abs(px - 3) + abs(py - 3)
    c2 = abs(px - 3) + abs(py - 4)
    c3 = abs(px - 4) + abs(py - 3)
    c4 = abs(px - 4) + abs(py - 4)
    return min(c1, min(c2, min(c3, c4)))


@njit(cache=True)
def fast_bfs_voronoi(sx: int, sy: int, ox: int, oy: int,
                     space_mask: uint64, primed_mask: uint64,
                     carpet_mask: uint64, blocked_mask: uint64):
    """
    BFS-based Voronoi + reach. Primed and blocked cells halt movement; the
    opposing worker blocks its own square. Returns
    (my_terr, opp_terr, my_reach, opp_reach).
    """
    INF = 127
    dist_me = np.full(64, INF, dtype=np.int16)
    dist_opp = np.full(64, INF, dtype=np.int16)
    q = np.empty(64, dtype=np.int16)

    opp_idx = oy * 8 + ox
    my_idx = sy * 8 + sx
    opp_bit = uint64(1) << uint64(opp_idx)
    my_bit = uint64(1) << uint64(my_idx)

    # BFS from me — opponent cell and all primed/blocked cells are walls
    qs, qe = 0, 1
    q[0] = my_idx
    dist_me[my_idx] = 0
    while qs < qe:
        idx = q[qs]
        qs += 1
        d = dist_me[idx]
        nd = d + 1
        x = idx % 8
        y = idx // 8
        if y > 0:
            nidx = idx - 8
            bit = uint64(1) << uint64(nidx)
            if nidx != opp_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_me[nidx] > nd:
                    dist_me[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if y < 7:
            nidx = idx + 8
            bit = uint64(1) << uint64(nidx)
            if nidx != opp_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_me[nidx] > nd:
                    dist_me[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if x > 0:
            nidx = idx - 1
            bit = uint64(1) << uint64(nidx)
            if nidx != opp_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_me[nidx] > nd:
                    dist_me[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if x < 7:
            nidx = idx + 1
            bit = uint64(1) << uint64(nidx)
            if nidx != opp_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_me[nidx] > nd:
                    dist_me[nidx] = nd
                    q[qe] = nidx
                    qe += 1

    # BFS from opponent
    qs, qe = 0, 1
    q[0] = opp_idx
    dist_opp[opp_idx] = 0
    while qs < qe:
        idx = q[qs]
        qs += 1
        d = dist_opp[idx]
        nd = d + 1
        x = idx % 8
        y = idx // 8
        if y > 0:
            nidx = idx - 8
            bit = uint64(1) << uint64(nidx)
            if nidx != my_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_opp[nidx] > nd:
                    dist_opp[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if y < 7:
            nidx = idx + 8
            bit = uint64(1) << uint64(nidx)
            if nidx != my_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_opp[nidx] > nd:
                    dist_opp[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if x > 0:
            nidx = idx - 1
            bit = uint64(1) << uint64(nidx)
            if nidx != my_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_opp[nidx] > nd:
                    dist_opp[nidx] = nd
                    q[qe] = nidx
                    qe += 1
        if x < 7:
            nidx = idx + 1
            bit = uint64(1) << uint64(nidx)
            if nidx != my_idx and not (blocked_mask & bit) and not (primed_mask & bit):
                if dist_opp[nidx] > nd:
                    dist_opp[nidx] = nd
                    q[qe] = nidx
                    qe += 1

    my_terr = 0.0
    opp_terr = 0.0
    my_reach = 0
    opp_reach = 0
    for idx in range(64):
        bit = uint64(1) << uint64(idx)
        if blocked_mask & bit:
            continue
        dm = dist_me[idx]
        do = dist_opp[idx]

        reachable_me = dm < INF
        reachable_opp = do < INF
        if reachable_me:
            my_reach += 1
        if reachable_opp:
            opp_reach += 1

        if not reachable_me and not reachable_opp:
            continue

        # Per-cell value: space and carpet contribute; primed walls don't
        if space_mask & bit:
            cell_val = 1.0
        elif carpet_mask & bit:
            cell_val = 1.2
        else:
            cell_val = 0.0

        if reachable_me and (not reachable_opp or dm < do):
            my_terr += cell_val
        elif reachable_opp and (not reachable_me or do < dm):
            opp_terr += cell_val
        else:
            my_terr += 0.15 * cell_val
            opp_terr += 0.15 * cell_val

    return my_terr, opp_terr, my_reach, opp_reach


@njit(cache=True)
def fast_open_reach(px: int, py: int, space_mask: uint64, primed_mask: uint64, carpet_mask: uint64) -> int:
    total = 0
    valid_mask = space_mask | primed_mask | carpet_mask
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for dx, dy in directions:
        cx, cy = px + dx, py + dy
        while 0 <= cx < 8 and 0 <= cy < 8:
            bit = uint64(1) << uint64(cy * 8 + cx)
            if (valid_mask & bit):
                total += 1
                cx += dx
                cy += dy
            else:
                break
    return total


@njit(cache=True)
def fast_adjacent_count(px: int, py: int, target_mask: uint64) -> int:
    count = 0
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for dx, dy in directions:
        nx, ny = px + dx, py + dy
        if 0 <= nx < 8 and 0 <= ny < 8:
            bit = uint64(1) << uint64(ny * 8 + nx)
            if (target_mask & bit):
                count += 1
    return count


@njit(cache=True)
def fast_longest_carpet_roll(px: int, py: int, primed_mask: uint64) -> int:
    best = 0
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for dx, dy in directions:
        cx, cy = px + dx, py + dy
        k = 0
        while 0 <= cx < 8 and 0 <= cy < 8:
            bit = uint64(1) << uint64(cy * 8 + cx)
            if (primed_mask & bit):
                k += 1
                cx += dx
                cy += dy
            else:
                break
        if k > best:
            best = k
    return best


# ==========================================
# AGENT DEFINITION
# ==========================================

# Precompute carpet points to avoid dictionary lookups in hot loops
CARPET_PTS = np.array([-1, -1, 2, 4, 6, 10, 15, 21], dtype=np.int32)

@dataclass
class _MCTSNode:
    board: object
    parent: "_MCTSNode | None"
    move: Move | None
    untried_moves: List[Move]
    prior: float = 0.0
    children: List["_MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class PlayerAgent:
    DIRS = {
        Direction.UP: (0, -1),
        Direction.RIGHT: (1, 0),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
    }

    def __init__(self, board, transition_matrix=None, time_left: Callable | None = None):
        self.turn = 0
        self.total_points = 0
        self._search_count = 0
        self._carpet_count = 0
        self._prime_count = 0
        self._plain_count = 0
        self._mcts_iters = 0
        self._nodes = 0

        self._rng = random.Random(0xC0FFEE)
        self._uct_c = 0.95
        self._prior_weight = 0.30

        self.rat_tracker = RatBeliefHMM(transition_matrix) if transition_matrix is not None else None
        self._belief_turn_count = 0

        # Per-turn cache of static evaluations (cleared at the top of each play()).
        self._eval_cache: dict = {}

        # Persistent MCTS tree root for tree reuse across turns.
        self._saved_root: Optional[_MCTSNode] = None
        self._saved_move: Optional[Move] = None

        self._warmup_numba()

    def _warmup_numba(self) -> None:
        u0 = np.uint64(0)
        fast_dist_to_center(3, 3)
        fast_bfs_voronoi(0, 0, 7, 7, u0, u0, u0, u0)
        fast_open_reach(3, 3, u0, u0, u0)
        fast_adjacent_count(3, 3, u0)
        fast_longest_carpet_roll(3, 3, u0)

    def commentate(self):
        return (
            f"Turns: {self.turn} | "
            f"Points: {self.total_points} | "
            f"MCTS iters: {self._mcts_iters} | "
            f"Nodes: {self._nodes}"
        )

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        self.total_points = board.player_worker.get_points()
        worker_pos = board.player_worker.get_location()

        # Fresh per-turn cache. Static eval depends on state signature, which is
        # consistent across turns — but the cache can grow unboundedly, so clear.
        self._eval_cache.clear()

        self._update_rat_belief(board, sensor_data, worker_pos)

        all_moves = board.get_valid_moves(exclude_search=False)
        if not all_moves:
            return None

        raw_board_moves = board.get_valid_moves(exclude_search=True)
        board_moves = self._filter_bad_carpets(raw_board_moves)

        if not board_moves:
            if self.rat_tracker is not None:
                self._search_count += 1
                self._saved_root = None
                self._saved_move = None
                return Move.search(self.rat_tracker.best_search_target())
            return random.choice(raw_board_moves if raw_board_moves else all_moves)

        best_board_move, best_board_value, iters = self._choose_board_move_mcts(board, board_moves, time_left)
        self._mcts_iters += iters

        if self.rat_tracker is not None and self._should_search(board, best_board_move, best_board_value):
            self._search_count += 1
            self._saved_root = None
            self._saved_move = None
            return Move.search(self.rat_tracker.best_search_target())

        if best_board_move.move_type == MoveType.CARPET:
            self._carpet_count += 1
        elif best_board_move.move_type == MoveType.PRIME:
            self._prime_count += 1
        elif best_board_move.move_type == MoveType.PLAIN:
            self._plain_count += 1

        self._saved_move = best_board_move
        return best_board_move

    def _update_rat_belief(self, board, sensor_data, worker_pos):
        if self.rat_tracker is None:
            return

        our_loc, our_hit = board.player_search
        if our_loc is not None:
            if our_hit:
                self.rat_tracker.reset_to_spawn()
            else:
                self.rat_tracker.apply_search_evidence(our_loc, False)

        is_first_a = self._belief_turn_count == 0 and board.is_player_a_turn
        if not is_first_a:
            self.rat_tracker.predict()
            opp_loc, opp_hit = board.opponent_search
            self.rat_tracker.apply_search_evidence(opp_loc, opp_hit)

        if sensor_data is not None:
            self.rat_tracker.update(board, sensor_data)

        self._belief_turn_count += 1

    def _should_search(self, board, best_board_move: Move, best_board_value: float) -> bool:
        target = self.rat_tracker.best_search_target()
        p_max = self.rat_tracker.get_belief_at(target)
        turns_left = board.player_worker.turns_left
        score_diff = board.player_worker.get_points() - board.opponent_worker.get_points()

        search_ev = 6.0 * p_max - 2.0
        board_ev = self._board_move_ev_estimate(board, best_board_move, best_board_value)

        if score_diff <= -10:
            threshold = 0.40
            margin = -0.50
        elif score_diff <= -6:
            threshold = 0.45
            margin = -0.25
        elif score_diff <= -2:
            threshold = 0.50
            margin = 0.0
        elif score_diff <= 2:
            threshold = 0.66
            margin = 0.45
        elif score_diff <= 6:
            threshold = 0.72
            margin = 0.70
        else:
            threshold = 0.78
            margin = 0.95

        if turns_left <= 8:
            if score_diff < 0:
                threshold -= 0.06
                margin -= 0.20
            else:
                threshold -= 0.02
        elif turns_left <= 14 and score_diff < 0:
            threshold -= 0.03

        _, _, entropy = self.rat_tracker.estimate()
        if entropy < 2.2:
            threshold -= 0.04
        elif entropy < 3.0:
            threshold -= 0.02

        threshold = min(0.88, max(0.40, threshold))
        return search_ev >= (board_ev + margin) and p_max >= threshold

    def _board_move_ev_estimate(self, board, move: Move, best_board_value: float) -> float:
        next_board = board.forecast_move(move)
        if next_board is None:
            return -math.inf
        immediate_gain = next_board.player_worker.get_points() - board.player_worker.get_points()
        # best_board_value is in [-1, 1] (tanh-bounded); scale back to a comparable point scale.
        long_term = 3.5 * best_board_value
        return immediate_gain + long_term

    def _choose_board_move_mcts(self, board, board_moves: List[Move], time_left: Callable) -> Tuple[Move, float, int]:
        ordered_root_moves = self._ordered_moves(board, board_moves)
        if len(ordered_root_moves) == 1:
            move = ordered_root_moves[0]
            self._saved_root = None
            return move, self._static_eval_after_my_move(board, move), 0

        # Immediate-win short-circuit: take any move that ends the game in our favour.
        for move in ordered_root_moves:
            forecast = board.forecast_move(move)
            if forecast is not None and forecast.is_game_over() and forecast.get_winner() == Result.PLAYER:
                self._saved_root = None
                return move, 1.0, 0

        remaining = max(time_left(), 0.0)
        budget = self._compute_turn_budget(remaining, board.player_worker.turns_left, len(ordered_root_moves))
        deadline = max(0.0, remaining - budget)

        root = self._build_root(board, ordered_root_moves)

        iterations = 0
        while time_left() > deadline:
            node = root

            while (not node.untried_moves) and node.children and (not node.board.is_game_over()):
                node = self._select_child(node)

            if node.untried_moves and (not node.board.is_game_over()):
                move = self._pop_untried_move(node.untried_moves)
                child_board = node.board.forecast_move(move)
                if child_board is None:
                    continue
                child_board.reverse_perspective()
                child_moves = self._ordered_moves(child_board, self._board_moves_for_sim(child_board))
                child = _MCTSNode(
                    board=child_board,
                    parent=node,
                    move=move,
                    untried_moves=child_moves,
                    prior=self._move_prior(node.board, move),
                )
                node.children.append(child)
                node = child

            value = self._rollout_value(node.board, time_left, deadline)
            self._backpropagate(node, value)

            # Once a node is fully expanded, it no longer needs its board reference.
            # This trims memory for deep trees.
            if not node.untried_moves and node.parent is not None:
                node.board = None

            iterations += 1

        if not root.children:
            best_move = ordered_root_moves[0]
            self._saved_root = None
            return best_move, self._static_eval_after_my_move(board, best_move), iterations

        # Robust child selection: most-visited with mean-value tiebreaker.
        best_child = max(root.children, key=lambda c: (c.visits, -c.mean_value))
        self._nodes += sum(child.visits for child in root.children)

        # Save root for tree reuse on the next turn.
        self._saved_root = root
        return best_child.move, -best_child.mean_value, iterations

    def _build_root(self, board, ordered_root_moves: List[Move]) -> _MCTSNode:
        """Either reuse a grandchild of the prior root or create a fresh root."""
        reused = self._try_reuse_tree(board, ordered_root_moves)
        if reused is not None:
            return reused

        return _MCTSNode(
            board=board.get_copy(False),
            parent=None,
            move=None,
            untried_moves=ordered_root_moves[:],
            prior=0.0,
        )

    def _try_reuse_tree(self, board, ordered_root_moves: List[Move]) -> Optional[_MCTSNode]:
        """
        If the saved tree contains a grandchild whose board matches `board`,
        promote it to a fresh root. Grandchildren already have our perspective
        (reverse_perspective is called twice), so they are directly comparable.
        """
        if self._saved_root is None or self._saved_move is None:
            return None

        saved_move = self._saved_move
        chosen_child = None
        for child in self._saved_root.children:
            if child.move is not None and self._moves_equal(child.move, saved_move):
                chosen_child = child
                break

        self._saved_root = None  # consumed either way
        self._saved_move = None
        if chosen_child is None or not chosen_child.children:
            return None

        target_key = self._board_key(board)
        matching = None
        for grandchild in chosen_child.children:
            gb = grandchild.board
            if gb is None:
                continue
            if self._board_key(gb) == target_key:
                matching = grandchild
                break

        if matching is None:
            return None

        # Detach from old parent and reset the promoted node as the new root.
        matching.parent = None
        matching.move = None
        # Reseed untried_moves in case the heuristic ordering changed since expansion.
        existing_move_keys = {self._move_key(c.move) for c in matching.children if c.move is not None}
        matching.untried_moves = [m for m in ordered_root_moves if self._move_key(m) not in existing_move_keys]
        return matching

    @staticmethod
    def _board_key(board) -> tuple:
        return (
            int(board._primed_mask),
            int(board._carpet_mask),
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.player_worker.get_points(),
            board.opponent_worker.get_points(),
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
        )

    @staticmethod
    def _move_key(move: Move) -> tuple:
        return (int(move.move_type), move.direction, move.roll_length, move.search_loc)

    @staticmethod
    def _moves_equal(a: Move, b: Move) -> bool:
        return (a.move_type == b.move_type and a.direction == b.direction
                and a.roll_length == b.roll_length and a.search_loc == b.search_loc)

    def _select_child(self, node: _MCTSNode) -> _MCTSNode:
        log_parent = math.log(node.visits + 1.0)
        best_score = -math.inf
        best_child = node.children[0]

        for child in node.children:
            if child.visits == 0:
                score = math.inf
            else:
                exploit = -child.mean_value
                explore = self._uct_c * math.sqrt(log_parent / child.visits)
                prior_bonus = self._prior_weight * child.prior / (1.0 + child.visits)
                score = exploit + explore + prior_bonus

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, node: _MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _pop_untried_move(self, moves: List[Move]) -> Move:
        if len(moves) > 2 and self._rng.random() < 0.15:
            return moves.pop(self._rng.randrange(min(3, len(moves))))
        return moves.pop(0)

    def _compute_turn_budget(self, remaining: float, turns_left: int, move_count: int) -> float:
        if remaining <= 0.2:
            return max(0.02, remaining * 0.7)

        base = remaining / (max(turns_left, 1) + 3)
        budget = 0.9 * base

        if move_count <= 6:
            budget *= 1.35

        # Endgame: commit banked time to the decisive moves instead of hoarding.
        if turns_left <= 3:
            budget = max(budget, remaining * 0.45)
        elif turns_left <= 8:
            budget *= 1.35
        elif turns_left <= 14:
            budget *= 1.1

        # Cap to protect against timeout losses but relax the ceiling when few turns remain.
        if turns_left <= 5:
            budget = min(budget, max(0.20, remaining - 0.30))
        else:
            budget = min(3.5, budget)
            budget = min(budget, max(0.05, remaining - 0.20))

        return max(0.10, budget)

    def _rollout_value(self, board, time_left: Callable, deadline: float) -> float:
        if board is None:
            return 0.0

        if board.is_game_over():
            return self._terminal_value(board)

        sim = board.get_copy(False)
        horizon = 10
        if sim.player_worker.turns_left <= 10:
            horizon = 14
        elif sim.player_worker.turns_left <= 20:
            horizon = 12

        for _ in range(horizon):
            if sim.is_game_over() or time_left() <= deadline:
                break
            moves = self._board_moves_for_sim(sim)
            if not moves:
                break
            move = self._rollout_policy(sim, moves)
            if not sim.apply_move(move, check_ok=False):
                break
            sim.reverse_perspective()

        if sim.is_game_over():
            return self._terminal_value(sim)
        return self._bounded_eval(sim)

    def _terminal_value(self, board) -> float:
        winner = board.get_winner()
        if winner == Result.PLAYER:
            return 1.0
        if winner == Result.ENEMY:
            return -1.0
        return 0.0

    def _bounded_eval(self, board) -> float:
        return math.tanh(self._evaluate(board) / 45.0)

    def _rollout_policy(self, board, moves: List[Move]) -> Move:
        ordered = self._ordered_moves(board, moves)
        if len(ordered) == 1:
            return ordered[0]

        # Opportunistic terminal-win pick during rollouts.
        top_k = min(3, len(ordered))
        for i in range(top_k):
            m = ordered[i]
            forecast = board.forecast_move(m)
            if forecast is not None and forecast.is_game_over() and forecast.get_winner() == Result.PLAYER:
                return m

        if self._rng.random() < 0.75:
            return ordered[0]
        return ordered[self._rng.randrange(top_k)]

    def _board_moves_for_sim(self, board) -> List[Move]:
        moves = board.get_valid_moves(exclude_search=True)
        filtered = self._filter_bad_carpets(moves)
        return filtered if filtered else moves

    def _filter_bad_carpets(self, moves: List[Move]) -> List[Move]:
        return [m for m in moves if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]

    def _ordered_moves(self, board, moves: List[Move]) -> List[Move]:
        return sorted(moves, key=lambda m: self._move_order_key(board, m), reverse=True)

    def _move_prior(self, board, move: Move) -> float:
        if move.move_type == MoveType.CARPET:
            pts = int(CARPET_PTS[min(move.roll_length, 7)])
            return min(1.0, 0.60 + 0.06 * max(0, pts))
        if move.move_type == MoveType.PRIME:
            return min(0.85, 0.40 + 0.02 * self._prime_local_value(board, move))
        if move.move_type == MoveType.PLAIN:
            return min(0.75, 0.28 + 0.02 * self._plain_local_value(board, move))
        return 0.0

    def _move_order_key(self, board, move: Move) -> float:
        if move.move_type == MoveType.CARPET:
            pts = int(CARPET_PTS[min(move.roll_length, 7)])
            # Prefer longer rolls; mild penalty for rolling far from the center early.
            return 1000.0 + 10.0 * pts + 0.5 * move.roll_length
        if move.move_type == MoveType.PRIME:
            return 500.0 + self._prime_local_value(board, move)
        if move.move_type == MoveType.PLAIN:
            return 100.0 + self._plain_local_value(board, move)
        return 0.0

    def _prime_local_value(self, board, move: Move) -> float:
        # Priming is most valuable next to existing primes (enables long rolls)
        # and near the center where future options converge.
        px, py = board.player_worker.get_location()
        primed_mask = np.uint64(board._primed_mask)
        neighbors = fast_adjacent_count(px, py, primed_mask)
        dist_c = fast_dist_to_center(px, py)
        return 5.0 + 3.0 * neighbors - 0.4 * dist_c

    def _plain_local_value(self, board, move: Move) -> float:
        dx, dy = self.DIRS[move.direction]
        px, py = board.player_worker.get_location()
        nx, ny = px + dx, py + dy
        primed_mask = np.uint64(board._primed_mask)
        # Encourage plain moves that set up a carpet roll from the destination.
        roll_bonus = fast_longest_carpet_roll(nx, ny, primed_mask)
        return -0.5 * fast_dist_to_center(nx, ny) + 1.2 * roll_bonus

    def _static_eval_after_my_move(self, board, move: Move) -> float:
        next_board = board.forecast_move(move)
        if next_board is None:
            return -1.0
        return self._bounded_eval(next_board)

    def _evaluate(self, board) -> float:
        me = board.player_worker
        opp = board.opponent_worker
        score_diff = me.get_points() - opp.get_points()

        if board.is_game_over():
            winner = board.get_winner()
            if winner == Result.PLAYER:
                return 1000.0 + score_diff
            if winner == Result.ENEMY:
                return -1000.0 + score_diff
            return float(score_diff)

        # Transposition lookup keyed on the full state signature.
        key = (
            int(board._primed_mask),
            int(board._carpet_mask),
            me.get_location(),
            opp.get_location(),
            me.get_points(),
            opp.get_points(),
            me.turns_left,
        )
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached

        sx, sy = me.get_location()
        ox, oy = opp.get_location()

        space = np.uint64(board._space_mask)
        primed = np.uint64(board._primed_mask)
        carpet = np.uint64(board._carpet_mask)
        blocked = np.uint64(board._blocked_mask)

        my_terr, opp_terr, my_reach, opp_reach = fast_bfs_voronoi(
            sx, sy, ox, oy, space, primed, carpet, blocked
        )

        my_roll = fast_longest_carpet_roll(sx, sy, primed)
        opp_roll = fast_longest_carpet_roll(ox, oy, primed)
        my_roll_pts = int(CARPET_PTS[min(my_roll, 7)]) if my_roll >= 2 else 0
        opp_roll_pts = int(CARPET_PTS[min(opp_roll, 7)]) if opp_roll >= 2 else 0

        turns_left = me.turns_left
        if turns_left > 25:
            center_w = 0.6
            score_w = 9.0
        elif turns_left > 12:
            center_w = 0.3
            score_w = 11.0
        else:
            center_w = 0.1
            score_w = 13.0
        my_center = fast_dist_to_center(sx, sy)

        # Mobility: count of legal move endpoints (ray-open squares) — a proxy
        # for how restricted each worker is by the evolving wall structure.
        my_open = fast_open_reach(sx, sy, space, primed, carpet)
        opp_open = fast_open_reach(ox, oy, space, primed, carpet)

        result = (
            score_w * score_diff
            + 1.5 * (my_terr - opp_terr)
            + 1.0 * (my_roll_pts - opp_roll_pts)
            + 0.35 * (my_reach - opp_reach)
            + 0.5 * (my_open - opp_open)
            - center_w * my_center
        )

        self._eval_cache[key] = result
        return result
