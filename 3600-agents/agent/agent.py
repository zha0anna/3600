from __future__ import annotations

from collections.abc import Callable
from typing import List, Optional, Tuple
import math
import random

from game.enums import BOARD_SIZE, CARPET_POINTS_TABLE, Cell, Direction, MoveType
from game.move import Move

from .rat_belief import RatBeliefHMM


class PlayerAgent:
    """
    Stronger hybrid agent:
    - exact HMM rat tracking
    - board evaluation with opponent-aware features
    - alpha-beta minimax over board moves
    - time-aware iterative deepening
    - explicit search-vs-board EV comparison
    """

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
        self._nodes = 0

        self.rat_tracker = RatBeliefHMM(transition_matrix) if transition_matrix is not None else None
        self._belief_turn_count = 0

    def commentate(self):
        return (
            f"Turns: {self.turn} | "
            f"Points: {self.total_points} | "
            f"Searches: {self._search_count} | "
            f"Carpets: {self._carpet_count} | "
            f"Primes: {self._prime_count} | "
            f"Plains: {self._plain_count} | "
            f"Nodes: {self._nodes}"
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
        board_moves = [m for m in board_moves if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]

        if not board_moves and self.rat_tracker is not None:
            self._search_count += 1
            return Move.search(self.rat_tracker.best_search_target())
        if not board_moves:
            return random.choice(moves)
        
        # --- GREEDY CARPET RULE ---
        best_carpet = None
        best_carpet_pts = -1

        for m in board_moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, -99)
                if pts > best_carpet_pts:
                    best_carpet_pts = pts
                    best_carpet = m

        # Always take strong carpets
        if best_carpet and best_carpet_pts >= 4:
            return best_carpet

        best_board_move, best_board_value = self._choose_board_move(board, board_moves, time_left)

        if self.rat_tracker is not None:
            turns_left = board.player_worker.turns_left
            target = self.rat_tracker.best_search_target()
            p_max = self.rat_tracker.get_belief_at(target)

            if p_max > 0.65 or (p_max > 0.5 and turns_left <= 8):
                self._search_count += 1
                return Move.search(target)

        if best_board_move.move_type == MoveType.CARPET:
            self._carpet_count += 1
        elif best_board_move.move_type == MoveType.PRIME:
            self._prime_count += 1
        elif best_board_move.move_type == MoveType.PLAIN:
            self._plain_count += 1
        return best_board_move

    def _update_rat_belief(self, board, sensor_data, worker_pos):
        if self.rat_tracker is None:
            return

        # Our own previous search: hit respawns the rat, miss zeros that square.
        our_loc, our_hit = board.player_search
        if our_loc is not None:
            if our_hit:
                self.rat_tracker.reset_to_spawn()
            else:
                self.rat_tracker.apply_search_evidence(our_loc, False)

        # Between our turns the rat moves twice (once before opponent, once before us).
        # Player A's very first turn is the exception: only one step since T_1000 init.
        is_first_a = (self._belief_turn_count == 0 and board.is_player_a_turn)
        if not is_first_a:
            self.rat_tracker.predict()
            opp_loc, opp_hit = board.opponent_search
            self.rat_tracker.apply_search_evidence(opp_loc, opp_hit)

        if sensor_data is not None:
            self.rat_tracker.update(board, sensor_data)

        self._belief_turn_count += 1

    def _choose_board_move(self, board, board_moves: List[Move], time_left: Callable) -> Tuple[Move, float]:
        ordered = sorted(board_moves, key=lambda m: self._move_order_key(board, m), reverse=True)
        remaining = max(time_left(), 0.0)
        turns_left = max(board.player_worker.turns_left, 1)

        # Light but safe time management under a 240s total budget.
        per_turn_budget = min(0.45, max(0.05, remaining / (turns_left + 10)))
        deadline = remaining - per_turn_budget

        best_move = ordered[0]
        best_value = -math.inf

        max_depth = 3
        if remaining > 45:
            max_depth = 3
        if remaining > 120 and len(ordered) <= 8:
            max_depth = 4

        for depth in range(1, max_depth + 1):
            if time_left() <= deadline:
                break

            current_best_move = best_move
            current_best_value = -math.inf
            alpha = -math.inf
            beta = math.inf

            for move in ordered:
                if time_left() <= deadline:
                    break
                next_board = board.forecast_move(move) #simulate the move to get the next board state
                if next_board is None:
                    continue
                next_board.reverse_perspective()
                value = self._alphabeta(next_board, depth - 1, alpha, beta, False, time_left, deadline)
                if value > current_best_value:
                    current_best_value = value
                    current_best_move = move
                alpha = max(alpha, current_best_value)

            if current_best_value > -math.inf:
                best_move = current_best_move
                best_value = current_best_value
                ordered.sort(key=lambda m: (m == best_move, self._move_order_key(board, m)), reverse=True)

        if best_value == -math.inf:
            best_value = self._static_eval_after_my_move(board, best_move)
        return best_move, best_value

    def _alphabeta(self, board, depth: int, alpha: float, beta: float, maximizing: bool,
                   time_left: Callable, deadline: float) -> float:
        self._nodes += 1
        if depth == 0 or board.is_game_over() or time_left() <= deadline:
            return self._evaluate(board)

        moves = [m for m in board.get_valid_moves(exclude_search=True)
                 if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if not moves:
            return self._evaluate(board)

        moves.sort(key=lambda m: self._move_order_key(board, m), reverse=True)

        if maximizing:
            value = -math.inf
            for move in moves:
                if time_left() <= deadline:
                    break
                next_board = board.forecast_move(move)
                if next_board is None:
                    continue
                next_board.reverse_perspective()
                value = max(value, self._alphabeta(next_board, depth - 1, alpha, beta, False, time_left, deadline))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = math.inf
        for move in moves:
            if time_left() <= deadline:
                break
            next_board = board.forecast_move(move)
            if next_board is None:
                continue
            next_board.reverse_perspective()
            value = min(value, self._alphabeta(next_board, depth - 1, alpha, beta, True, time_left, deadline))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _move_order_key(self, board, move: Move) -> float:
        if move.move_type == MoveType.CARPET:
            return 1000 + 30 * CARPET_POINTS_TABLE.get(move.roll_length, -10) + move.roll_length
        if move.move_type == MoveType.PRIME:
            return 200 + self._prime_local_value(board, move)
        if move.move_type == MoveType.PLAIN:
            return 50 + self._plain_local_value(board, move)
        return -999

    def _evaluate(self, board) -> float: #evaluate the board state from the perspective of the current player, higher is better
        me = board.player_worker
        opp = board.opponent_worker
        my_pos = me.get_location()
        opp_pos = opp.get_location()

        my_points = me.get_points()
        opp_points = opp.get_points()
        score_diff = my_points - opp_points

        my_immediate = self._best_carpet_points(board, enemy=False)
        opp_immediate = self._best_carpet_points(board, enemy=True)

        my_prime_potential = self._best_prime_setup(board, enemy=False)
        opp_prime_potential = self._best_prime_setup(board, enemy=True)

        my_mobility = len(board.get_valid_moves(enemy=False, exclude_search=True))
        opp_mobility = len(board.get_valid_moves(enemy=True, exclude_search=True))

        my_territory = self._territory_value(board, my_pos, opp_pos)
        opp_territory = self._territory_value(board, opp_pos, my_pos)

        center_bonus = -0.15 * self._dist_to_center(my_pos) + 0.15 * self._dist_to_center(opp_pos)
        intercept = -0.15 * self._manhattan(my_pos, opp_pos)

        return (
            12.0 * score_diff
            + 7.5 * my_immediate
            - 9.0 * opp_immediate
            + 2.5 * (my_prime_potential - opp_prime_potential)
            + 0.25 * (my_mobility - opp_mobility)
            + 0.30 * (my_territory - opp_territory)
            + center_bonus
            + intercept
        )

    def _static_eval_after_my_move(self, board, move: Move) -> float:
        next_board = board.forecast_move(move)
        if next_board is None:
            return -math.inf
        next_board.reverse_perspective()
        return -self._evaluate(next_board)

    def _best_carpet_points(self, board, enemy: bool = False) -> int:
        moves = board.get_valid_moves(enemy=enemy, exclude_search=True)
        best = 0
        for m in moves:
            if m.move_type == MoveType.CARPET:
                best = max(best, CARPET_POINTS_TABLE.get(m.roll_length, 0))
        return best

    def _best_prime_setup(self, board, enemy: bool = False) -> float:
        moves = board.get_valid_moves(enemy=enemy, exclude_search=True)
        worker = board.opponent_worker if enemy else board.player_worker
        pos = worker.get_location()
        best = 0.0
        for m in moves:
            if m.move_type != MoveType.PRIME:
                continue
            existing = self._count_primed_in_direction(pos, m.direction, board)
            open_space = self._open_squares_in_direction(pos, m.direction, board)
            future_len = max(2, existing + 1)
            future_pts = CARPET_POINTS_TABLE.get(min(future_len, 7), 21)
            best = max(best, future_pts + 0.75 * open_space)
        return best

    def _prime_local_value(self, board, move: Move) -> float:
        pos = board.player_worker.get_location()
        existing = self._count_primed_in_direction(pos, move.direction, board)
        open_space = self._open_squares_in_direction(pos, move.direction, board)
        future_len = max(2, existing + 1)
        future_pts = CARPET_POINTS_TABLE.get(min(future_len, 7), 21)
        dest = self._dest_after_move(pos, move.direction)
        return 5.0 * future_pts + 2.0 * open_space + 0.4 * self._territory_value(board, dest, board.opponent_worker.get_location())

    def _plain_local_value(self, board, move: Move) -> float:
        pos = board.player_worker.get_location()
        dest = self._dest_after_move(pos, move.direction)
        best_reachable = self._best_carpet_from(dest, board)
        open_reach = self._total_open_reach(dest, board)
        adj_primed = self._adjacent_primed_count(dest, board)
        return 3.0 * best_reachable + 0.8 * open_reach + 1.2 * adj_primed - 1.5 * self._dist_to_center(dest)

    def _best_carpet_from(self, pos, board) -> int:
        best = 0
        for direction in Direction:
            run_len = self._count_primed_in_direction(pos, direction, board)
            if run_len >= 2:
                best = max(best, CARPET_POINTS_TABLE.get(min(run_len, 7), 21))
        return best

    def _count_primed_in_direction(self, pos, direction, board) -> int:
        dx, dy = self.DIRS[direction]
        x, y = pos
        count = 0
        cx, cy = x + dx, y + dy
        for _ in range(BOARD_SIZE - 1):
            if not (0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE):
                break
            if board.get_cell((cx, cy)) == Cell.PRIMED:
                count += 1
                cx += dx
                cy += dy
            else:
                break
        return count

    def _open_squares_in_direction(self, pos, direction, board) -> int:
        dx, dy = self.DIRS[direction]
        x, y = pos
        cx, cy = x + dx, y + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE and board.get_cell((cx, cy)) == Cell.PRIMED:
            cx += dx
            cy += dy
        count = 0
        for _ in range(BOARD_SIZE - 1):
            if not (0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE):
                break
            if board.get_cell((cx, cy)) == Cell.SPACE:
                count += 1
                cx += dx
                cy += dy
            else:
                break
        return count

    def _adjacent_primed_count(self, pos, board) -> int:
        x, y = pos
        count = 0
        for dx, dy in self.DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board.get_cell((nx, ny)) == Cell.PRIMED:
                count += 1
        return count

    def _total_open_reach(self, pos, board) -> int:
        total = 0
        x, y = pos
        for dx, dy in self.DIRS.values():
            cx, cy = x + dx, y + dy
            for _ in range(BOARD_SIZE - 1):
                if not (0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE):
                    break
                cell = board.get_cell((cx, cy))
                if cell in (Cell.SPACE, Cell.PRIMED, Cell.CARPET):
                    total += 1
                    cx += dx
                    cy += dy
                else:
                    break
        return total

    def _territory_value(self, board, src, other) -> float:
        sx, sy = src
        ox, oy = other
        total = 0.0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                loc = (x, y)
                cell = board.get_cell(loc)
                if cell == Cell.BLOCKED:
                    continue
                d_me = abs(sx - x) + abs(sy - y)
                d_opp = abs(ox - x) + abs(oy - y)
                if d_me < d_opp:
                    total += 1.0
                    if cell == Cell.SPACE:
                        total += 0.35
                    elif cell == Cell.PRIMED:
                        total += 0.20
                elif d_me == d_opp:
                    total += 0.15
        return total

    def _dist_to_center(self, pos) -> int:
        centers = [(3, 3), (3, 4), (4, 3), (4, 4)]
        return min(abs(pos[0] - cx) + abs(pos[1] - cy) for cx, cy in centers)

    def _dest_after_move(self, pos, direction) -> Tuple[int, int]:
        dx, dy = self.DIRS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def _manhattan(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])