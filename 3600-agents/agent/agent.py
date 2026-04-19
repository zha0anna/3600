from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Tuple
import math
import random

from game.enums import BOARD_SIZE, CARPET_POINTS_TABLE, Cell, Direction, MoveType
from game.move import Move
from game.worker import Worker

from .rat_belief import RatBeliefHMM


# Transposition-table flags
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 64-bit mask
ALL_MASK = 0xFFFFFFFFFFFFFFFF


class PlayerAgent:
    """
    Highly Optimized Hybrid Agent:
    - O(1) Bitboard Territory Evaluation
    - Lightweight Ray-casting for mobility/potential
    - Alpha-beta minimax with iterative deepening + TT (EXACT/LOWER/UPPER bounds)
    - Killer-move heuristic and aspiration windows
    - Depth-aware terminal states (prefers faster wins)
    """

    DIRS = {
        Direction.UP: (0, -1),
        Direction.RIGHT: (1, 0),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
    }
    DIR_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    TIME_PER_TURN_CAP = 8.0
    TIME_PER_TURN_FLOOR = 0.20
    TIME_USE_FRACTION = 0.85
    SAFETY_BUFFER = 5.0
    MAX_ITERATIVE_DEPTH = 24
    TIME_CHECK_PERIOD = 512
    ASPIRATION_WINDOW = 8.0

    def __init__(self, board, transition_matrix=None, time_left: Callable | None = None):
        self.turn = 0
        self.total_points = 0
        self._search_count = 0
        self._carpet_count = 0
        self._prime_count = 0
        self._plain_count = 0
        self._nodes = 0
        self._last_depth_reached = 0

        self.rat_tracker = RatBeliefHMM(transition_matrix) if transition_matrix is not None else None
        self._belief_turn_count = 0

        self._tt: Dict[tuple, Tuple[int, float, int, Optional[tuple]]] = {}
        self._killers: Dict[int, Tuple[Optional[tuple], Optional[tuple]]] = {}

        self._init_bitboards()

    def _init_bitboards(self):
        self.t_closer = [[0]*64 for _ in range(64)]
        self.t_equal = [[0]*64 for _ in range(64)]
        for p1 in range(64):
            x1, y1 = p1 % 8, p1 // 8
            for p2 in range(64):
                x2, y2 = p2 % 8, p2 // 8
                c_mask = 0
                e_mask = 0
                for c in range(64):
                    cx, cy = c % 8, c // 8
                    d1 = abs(x1 - cx) + abs(y1 - cy)
                    d2 = abs(x2 - cx) + abs(y2 - cy)
                    if d1 < d2:
                        c_mask |= (1 << c)
                    elif d1 == d2:
                        e_mask |= (1 << c)
                self.t_closer[p1][p2] = c_mask
                self.t_equal[p1][p2] = e_mask

    def commentate(self):
        return (
            f"Turns: {self.turn} | Points: {self.total_points} | "
            f"Nodes: {self._nodes} | LastDepth: {self._last_depth_reached}"
        )

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        self.total_points = board.player_worker.get_points()
        worker_pos = board.player_worker.get_location()

        self._update_rat_belief(board, sensor_data, worker_pos)

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        board_moves = [m for m in moves if m.move_type != MoveType.SEARCH]

        if not board_moves and self.rat_tracker is not None:
            self._search_count += 1
            return Move.search(self.rat_tracker.best_search_target())
        if not board_moves:
            return random.choice(moves)
        
        #carpet greed rule
        best_carpet = None
        best_pts = -1

        for m in board_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best_pts:
                    best_pts = pts
                    best_carpet = m

        if best_carpet and best_pts >= 6:
            return best_carpet

        best_board_move, best_board_value, best_immediate = self._choose_board_move(
            board, board_moves, time_left
        )

        if self.rat_tracker is not None:
            turns_left = board.player_worker.turns_left
            target = self.rat_tracker.best_search_target()
            p_max = self.rat_tracker.get_belief_at(target)
            search_ev = 6.0 * p_max - 2.0

            margin = 1.0
            if turns_left <= 6: margin = 0.0 #encourage searching at end of game
            elif turns_left <= 12: margin = 0.5

            if search_ev > best_immediate + margin or p_max > 0.6:
                self._search_count += 1
                return Move.search(target)

        if best_board_move.move_type == MoveType.CARPET: self._carpet_count += 1
        elif best_board_move.move_type == MoveType.PRIME: self._prime_count += 1
        elif best_board_move.move_type == MoveType.PLAIN: self._plain_count += 1

        return best_board_move

    def _update_rat_belief(self, board, sensor_data, worker_pos):
        if self.rat_tracker is None: return

        our_loc, our_hit = board.player_search
        if our_loc is not None:
            if our_hit: self.rat_tracker.reset_to_spawn()
            else: self.rat_tracker.apply_search_evidence(our_loc, False)

        if not (self._belief_turn_count == 0 and board.is_player_a_turn):
            self.rat_tracker.predict()
            opp_loc, opp_hit = board.opponent_search
            self.rat_tracker.apply_search_evidence(opp_loc, opp_hit)

        if sensor_data is not None:
            self.rat_tracker.update(board, sensor_data)

        self._belief_turn_count += 1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _move_key(self, move: Move) -> tuple:
        return (int(move.move_type),
                int(move.direction) if move.direction is not None else -1,
                move.roll_length, move.search_loc)

    def _store_killer(self, depth: int, move_key: tuple) -> None:
        k1, k2 = self._killers.get(depth, (None, None))
        if move_key == k1: return
        self._killers[depth] = (move_key, k1)

    def _choose_board_move(self, board, board_moves: List[Move], time_left: Callable) -> Tuple[Move, float, int]:
        ordered = sorted(board_moves, key=lambda m: self._move_order_key(board, m), reverse=True)
        remaining = max(time_left(), 0.0)
        turns_left = max(board.player_worker.turns_left, 1)

        budget = (remaining - self.SAFETY_BUFFER) / turns_left * self.TIME_USE_FRACTION
        per_turn_budget = min(self.TIME_PER_TURN_CAP, max(self.TIME_PER_TURN_FLOOR, budget))
        deadline = remaining - per_turn_budget

        self._tt.clear()
        self._killers.clear()
        best_move = ordered[0]
        best_value = -math.inf
        self._last_depth_reached = 0

        for depth in range(1, self.MAX_ITERATIVE_DEPTH + 1):
            if time_left() <= deadline: break

            # Safely guard the aspiration window to avoid terminal scores
            use_aspiration = depth >= 3 and best_value > -99000.0 and best_value < 99000.0
            if use_aspiration:
                aw_low = best_value - self.ASPIRATION_WINDOW
                aw_high = best_value + self.ASPIRATION_WINDOW
            else:
                aw_low = -math.inf
                aw_high = math.inf

            current_best_move = best_move
            current_best_value = -math.inf
            timed_out = False

            for attempt in range(3):
                current_best_move = best_move
                current_best_value = -math.inf
                alpha = aw_low
                beta = aw_high
                timed_out = False

                try:
                    for move in ordered:
                        if (self._nodes & (self.TIME_CHECK_PERIOD - 1)) == 0 and time_left() <= deadline:
                            raise TimeoutError()

                        next_board = board.forecast_move(move)
                        if next_board is None: continue
                        next_board.reverse_perspective()

                        value = self._alphabeta(next_board, depth - 1, alpha, beta, False, time_left, deadline)

                        if value > current_best_value:
                            current_best_value = value
                            current_best_move = move
                        if current_best_value > alpha: alpha = current_best_value
                except TimeoutError:
                    timed_out = True
                    break

                # Aspiration retry: widen on fail-low / fail-high
                if current_best_value <= aw_low and aw_low != -math.inf:
                    aw_low = -math.inf
                    continue
                if current_best_value >= aw_high and aw_high != math.inf:
                    aw_high = math.inf
                    continue
                break

            if not timed_out and current_best_value > -math.inf:
                best_move = current_best_move
                best_value = current_best_value
                self._last_depth_reached = depth
                ordered = [best_move] + [m for m in ordered if m is not best_move]
            else:
                break

        if best_value == -math.inf:
            best_value = self._static_eval_after_my_move(board, best_move)

        return best_move, best_value, self._immediate_value(best_move)

    def _alphabeta(self, board, depth: int, alpha: float, beta: float, maximizing: bool,
                   time_left: Callable, deadline: float) -> float:
        self._nodes += 1

        if (self._nodes & (self.TIME_CHECK_PERIOD - 1)) == 0 and time_left() <= deadline:
            raise TimeoutError()

        # Depth-aware termination rewards faster wins
        if board.is_game_over():
            diff = board.player_worker.points - board.opponent_worker.points
            if diff > 0: return 99999.0 + depth if maximizing else -99999.0 - depth
            if diff < 0: return -99999.0 - depth if maximizing else 99999.0 + depth
            return 0.0

        if depth == 0:
            return self._evaluate(board) if maximizing else -self._evaluate(board)

        # Uses get_location() instead of position to avoid AttributeError
        tt_key = (board._primed_mask, board._carpet_mask,
                  board.player_worker.get_location(), board.opponent_worker.get_location(),
                  board.player_worker.points - board.opponent_worker.points,
                  board.player_worker.turns_left, maximizing)

        original_alpha, original_beta = alpha, beta
        tt_move_key = None

        cached = self._tt.get(tt_key)
        if cached is not None:
            cached_depth, cached_value, cached_flag, tt_move_key = cached
            if cached_depth >= depth:
                if cached_flag == TT_EXACT: return cached_value
                if cached_flag == TT_LOWER and cached_value >= beta: return cached_value
                if cached_flag == TT_UPPER and cached_value <= alpha: return cached_value
                if cached_flag == TT_LOWER and cached_value > alpha: alpha = cached_value
                elif cached_flag == TT_UPPER and cached_value < beta: beta = cached_value
                if alpha >= beta: return cached_value

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            v = self._evaluate(board) if maximizing else -self._evaluate(board)
            self._tt[tt_key] = (depth, v, TT_EXACT, None)
            return v

        # Cheap type-based ordering for inner nodes (full ray-walk ordering only at root).
        moves.sort(key=self._quick_move_order, reverse=True)

        # Insert TT move at position 0
        if tt_move_key is not None:
            for i in range(len(moves)):
                if self._move_key(moves[i]) == tt_move_key:
                    if i != 0: moves.insert(0, moves.pop(i))
                    break

        # Insert killer moves right after TT move
        k1, k2 = self._killers.get(depth, (None, None))
        insert_pos = 1 if tt_move_key is not None else 0
        for killer in (k1, k2):
            if killer is None or killer == tt_move_key: continue
            for i in range(insert_pos, len(moves)):
                if self._move_key(moves[i]) == killer:
                    if i != insert_pos:
                        moves.insert(insert_pos, moves.pop(i))
                    insert_pos += 1
                    break

        best_local_move = None

        if maximizing:
            value = -math.inf
            for move in moves:
                next_board = board.forecast_move(move)
                if next_board is None: continue
                next_board.reverse_perspective()

                child = self._alphabeta(next_board, depth - 1, alpha, beta, False, time_left, deadline)
                if child > value:
                    value = child
                    best_local_move = move
                if value > alpha: alpha = value
                if alpha >= beta:
                    self._store_killer(depth, self._move_key(move))
                    break
        else:
            value = math.inf
            for move in moves:
                next_board = board.forecast_move(move)
                if next_board is None: continue
                next_board.reverse_perspective()

                child = self._alphabeta(next_board, depth - 1, alpha, beta, True, time_left, deadline)
                if child < value:
                    value = child
                    best_local_move = move
                if value < beta: beta = value
                if alpha >= beta:
                    self._store_killer(depth, self._move_key(move))
                    break

        flag = TT_UPPER if value <= original_alpha else TT_LOWER if value >= original_beta else TT_EXACT
        self._tt[tt_key] = (depth, value, flag, self._move_key(best_local_move) if best_local_move else tt_move_key)
        return value

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _quick_move_order(self, move: Move) -> float:
        mt = move.move_type
        if mt == MoveType.CARPET:
            rl = move.roll_length
            if rl == 1: return -50.0
            return 1000.0 + 30.0 * CARPET_POINTS_TABLE.get(rl, -10) + rl
        if mt == MoveType.PRIME: return 200.0
        if mt == MoveType.PLAIN: return 50.0
        return -999.0

    def _move_order_key(self, board, move: Move) -> float:
        if move.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(move.roll_length, -10)
            return -50 if move.roll_length == 1 else 1000 + 30 * pts + move.roll_length
        if move.move_type == MoveType.PRIME:
            return 200 + self._prime_local_value(board, move)
        if move.move_type == MoveType.PLAIN:
            return 50 + self._plain_local_value(board, move)
        return -999

    def _immediate_value(self, move: Move) -> int:
        if move.move_type == MoveType.CARPET: return CARPET_POINTS_TABLE.get(move.roll_length, 0)
        if move.move_type == MoveType.PRIME: return 1
        return 0

    def _prime_local_value(self, board, move: Move) -> float:
        pos = board.player_worker.get_location()
        dx, dy = self.DIRS[move.direction]
        cx, cy = pos[0] + dx, pos[1] + dy
        primed_count = 0

        while 0 <= cx < 8 and 0 <= cy < 8:
            if board._primed_mask & (1 << (cy * 8 + cx)):
                primed_count += 1
                cx += dx; cy += dy
            else: break

        future_pts = CARPET_POINTS_TABLE.get(min(max(2, primed_count + 1), 7), 21)
        dest = (pos[0] + dx, pos[1] + dy)
        territory = self._fast_territory(board, dest, board.opponent_worker.get_location())
        return 5.0 * future_pts + 0.4 * territory

    def _plain_local_value(self, board, move: Move) -> float:
        pos = board.player_worker.get_location()
        dx, dy = self.DIRS[move.direction]
        dest = (pos[0] + dx, pos[1] + dy)

        adj_primed = 0
        for ndx, ndy in self.DIRS.values():
            nx, ny = dest[0] + ndx, dest[1] + ndy
            if 0 <= nx < 8 and 0 <= ny < 8 and (board._primed_mask & (1 << (ny * 8 + nx))):
                adj_primed += 1

        best_reachable = 0
        for direction in Direction:
            ddx, ddy = self.DIRS[direction]
            cx, cy = dest[0] + ddx, dest[1] + ddy
            p_count = 0
            while 0 <= cx < 8 and 0 <= cy < 8:
                if board._primed_mask & (1 << (cy * 8 + cx)):
                    p_count += 1
                    cx += ddx; cy += ddy
                else: break
            if p_count >= 2:
                best_reachable = max(best_reachable, CARPET_POINTS_TABLE.get(min(p_count, 7), 21))

        return 3.0 * best_reachable + 1.2 * adj_primed - 1.5 * self._dist_to_center(dest)

    # ------------------------------------------------------------------
    # Optimized Evaluation (No Object Generation)
    # ------------------------------------------------------------------

    def _fast_eval_worker(self, pos, opp_pos, board) -> Tuple[int, int, int]:
        mobility = best_carpet = best_prime = 0
        x, y = pos
        # Cells unwalkable for plain/prime move-into: blocked OR primed.
        impassable = board._blocked_mask | board._primed_mask

        for dx, dy in self.DIR_DELTAS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 8 and 0 <= ny < 8): continue

            idx = ny * 8 + nx
            bit = 1 << idx
            # Walking into a primed cell isn't a legal plain move; carpet ray starts here.
            walkable = not (impassable & bit) and not (nx == opp_pos[0] and ny == opp_pos[1])
            if walkable:
                mobility += 1

            primed_count = 0
            cx, cy = nx, ny
            while 0 <= cx < 8 and 0 <= cy < 8:
                if board._primed_mask & (1 << (cy * 8 + cx)):
                    primed_count += 1
                    cx += dx; cy += dy
                else: break

            if primed_count >= 2:
                pts = CARPET_POINTS_TABLE.get(min(primed_count, 7), 21)
                if pts > best_carpet: best_carpet = pts

            future_pts = CARPET_POINTS_TABLE.get(min(primed_count + 1, 7), 21)
            if future_pts > best_prime: best_prime = future_pts

        return mobility, best_carpet, best_prime

    def _fast_territory(self, board, p1_pos, p2_pos) -> float:
        p1 = p1_pos[1] * 8 + p1_pos[0]
        p2 = p2_pos[1] * 8 + p2_pos[0]

        valid_mask = ~board._blocked_mask
        c_mask = self.t_closer[p1][p2] & valid_mask
        e_mask = self.t_equal[p1][p2] & valid_mask

        c_spaces = (c_mask & board._space_mask).bit_count()
        c_primed = (c_mask & board._primed_mask).bit_count()
        e_spaces = (e_mask & board._space_mask).bit_count()
        e_primed = (e_mask & board._primed_mask).bit_count()

        return (c_mask.bit_count() + c_spaces * 0.35 + c_primed * 0.20 +
                e_mask.bit_count() * 0.15 + e_spaces * 0.05 + e_primed * 0.03)

    def _evaluate(self, board) -> float:
        my_pos = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        score_diff = board.player_worker.get_points() - board.opponent_worker.get_points()

        my_mob, my_imm_carp, my_prime_pot = self._fast_eval_worker(my_pos, opp_pos, board)
        opp_mob, opp_imm_carp, opp_prime_pot = self._fast_eval_worker(opp_pos, my_pos, board)

        trap_pen = 0.0
        if my_mob <= 1: trap_pen -= 6.0
        elif my_mob <= 2: trap_pen -= 2.0
        if opp_mob <= 1: trap_pen += 6.0
        elif opp_mob <= 2: trap_pen += 2.0

        my_territory = self._fast_territory(board, my_pos, opp_pos)
        opp_territory = self._fast_territory(board, opp_pos, my_pos)

        center_bonus = -0.15 * self._dist_to_center(my_pos) + 0.15 * self._dist_to_center(opp_pos)
        intercept = -0.10 * self._manhattan(my_pos, opp_pos)

        return (
            12.0 * score_diff
            + 7.5 * my_imm_carp
            - 9.0 * opp_imm_carp
            + 2.5 * (my_prime_pot - opp_prime_pot)
            + 0.50 * (my_mob - opp_mob)
            + 0.30 * (my_territory - opp_territory)
            + center_bonus
            + intercept
            + trap_pen
        )

    def _static_eval_after_my_move(self, board, move: Move) -> float:
        next_board = board.forecast_move(move)
        if next_board is None: return -math.inf
        next_board.reverse_perspective()
        return -self._evaluate(next_board)

    def _dist_to_center(self, pos) -> int:
        return min(abs(pos[0] - cx) + abs(pos[1] - cy) for cx, cy in [(3, 3), (3, 4), (4, 3), (4, 4)])

    def _manhattan(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])