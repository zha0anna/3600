from collections.abc import Callable
from typing import List, Tuple
import random

from game import board, move, enums
from game.enums import MoveType, Direction, CARPET_POINTS_TABLE, Cell, BOARD_SIZE
from game.move import Move

from .rat_tracker import RatTracker


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0
        self.total_points = 0
        self._search_count = 0
        self._carpet_count = 0
        self._prime_count = 0
        self._plain_count = 0

        if transition_matrix is not None:
            self.rat_tracker = RatTracker(transition_matrix, board)
        else:
            self.rat_tracker = None

    def commentate(self):
        return (
        f"Turns: {self.turn} | "
        f"Points: {self.total_points} | "
        f"Searches: {self._search_count} | "
        f"Carpets: {self._carpet_count} | "
        f"Primes: {self._prime_count} | "
        f"Plains: {self._plain_count}"
    )

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        self.total_points = board.player_worker.get_points()
        worker_pos = board.player_worker.get_location()

        # --- Handle opponent's last search result ---
        # board.opponent_search = (search_loc, was_correct)
        # Use this to update our belief even on the opponent's turn
        if self.rat_tracker is not None:
            opp_loc, opp_hit = board.opponent_search
            if opp_loc is not None:
                if opp_hit:
                    # Rat was caught — new rat spawned, reset belief entirely
                    self.rat_tracker.record_hit(board)
                else:
                    # Opponent missed — zero out that cell
                    self.rat_tracker.record_opponent_miss(opp_loc)

        # --- Update rat belief with this turn's sensor data ---
        if self.rat_tracker is not None and sensor_data is not None:
            self.rat_tracker.update(sensor_data, worker_pos, board)

        # --- Handle our own last search result ---
        # board.player_search = (search_loc, was_correct)
        if self.rat_tracker is not None:
            our_loc, our_hit = board.player_search
            if our_loc is not None:
                if our_hit:
                    self.rat_tracker.record_hit(board)
                else:
                    self.rat_tracker.record_our_miss(our_loc)

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        # --- Separate board moves from search moves ---
        board_moves = [m for m in moves if m.move_type != MoveType.SEARCH]

        # Filter out single-square carpets — they cost -1 point
        board_moves = [
            m for m in board_moves
            if not (m.move_type == MoveType.CARPET and m.roll_length == 1)
        ]

        # --- Decide whether to search for the rat ---
        if self.rat_tracker is not None:
            turns_left = board.player_worker.turns_left
            best_carpet_pts = self._best_carpet_points(moves)
            
            # Late game: lower the carpet threshold so we search even if a medium carpet exists
            if turns_left <= 8:
                carpet_threshold = 10   # only skip search for rolls of 5+
            elif turns_left <= 15:
                carpet_threshold = 8    # only skip for rolls of 4+  
            else:
                carpet_threshold = 6    # default: skip for rolls of 4+

            if self.rat_tracker.should_search(turns_left) and best_carpet_pts < carpet_threshold:
                self._search_count += 1
                target = self.rat_tracker.best_search_target()
                return Move.search(target)

        # --- Pick best board move ---
        if not board_moves:
            fallback = [m for m in moves if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
            return random.choice(fallback) if fallback else random.choice(moves)

        best_move, best_score = None, float('-inf')
        for m in board_moves:
            s = self._score_move(m, board)
            if s > best_score:
                best_score = s
                best_move = m

        if best_move.move_type == MoveType.CARPET:
            self._carpet_count = getattr(self, '_carpet_count', 0) + 1
        elif best_move.move_type == MoveType.PRIME:
            self._prime_count = getattr(self, '_prime_count', 0) + 1
        elif best_move.move_type == MoveType.PLAIN:
            self._plain_count = getattr(self, '_plain_count', 0) + 1

        return best_move

    # -----------------------------------------------------------------------
    # MOVE SCORING
    # -----------------------------------------------------------------------

    def _score_move(self, m, board) -> float:
        pos = board.player_worker.get_location()

        # --- CARPET: immediate points, always top priority ---
        # We can carpet ANY primed squares, including opponent's
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, -99)
            if pts < 2:
                return -50  # never carpet a single square (-1 pts)
            return pts * 10 + m.roll_length * 5

        # --- PRIME: score by future carpet payoff in that direction ---
        if m.move_type == MoveType.PRIME:
            existing = self._count_primed_in_direction(pos, m.direction, board)
            open_space = self._open_squares_in_direction(pos, m.direction, board)
            
            # Clamp to 2 minimum — priming always creates at least potential for
            # a length-2 carpet (this square + one more prime next turn)
            future_carpet_len = max(existing + 1, 2)
            future_pts = CARPET_POINTS_TABLE.get(min(future_carpet_len, 7), 21)
            
            # Open space is very valuable — it means we can keep extending
            return future_pts * 8 + open_space * 6 + 10

        if m.move_type == MoveType.PLAIN:
            dest = self._dest_after_move(pos, m.direction)
    
            # What can we carpet immediately from dest?
            best_reachable = self._best_carpet_from(dest, board)
            
            # How much open territory exists around dest for future priming?
            open_reach = self._total_open_reach(dest, board)
            
            # How many primed neighbors does dest have?
            adj_primed = self._adjacent_primed_count(dest, board)
            
            # Only worth moving if destination is genuinely productive
            # High penalty keeps plains as last resort
            score = best_reachable * 8 + open_reach * 1.5 + adj_primed * 4 - 50
            return score

        return -100

    def _best_carpet_points(self, moves) -> int:
        """Return the highest carpet points available this turn, or 0."""
        best = 0
        for m in moves:
            if m.move_type == MoveType.CARPET:
                pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
                if pts > best:
                    best = pts
        return best

    def _best_carpet_from(self, pos, board) -> int:
        """
        Best carpet roll reachable from pos in any direction.
        Counts all primed squares regardless of who placed them.
        """
        best = 0
        for direction in Direction:
            run_len = self._count_primed_in_direction(pos, direction, board)
            if run_len >= 2:
                pts = CARPET_POINTS_TABLE.get(min(run_len, 7), 21)
                if pts > best:
                    best = pts
        return best

    # -----------------------------------------------------------------------
    # DIRECTIONAL HELPERS
    # -----------------------------------------------------------------------

    DIRS = {
        Direction.UP:    (0, -1),
        Direction.DOWN:  (0,  1),
        Direction.LEFT:  (-1, 0),
        Direction.RIGHT: ( 1, 0),
    }

    def _dest_after_move(self, pos, direction) -> Tuple:
        dx, dy = self.DIRS[direction]
        return (pos[0] + dx, pos[1] + dy)

    def _count_primed_in_direction(self, pos, direction, board) -> int:
        """
        Count contiguous primed squares starting from the square immediately
        ahead in the given direction. Counts all primed squares regardless
        of who placed them — opponent lines are carpetable too.
        """
        dx, dy = self.DIRS[direction]
        x, y = pos
        count = 0
        cx, cy = x + dx, y + dy
        for _ in range(7):
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
        """Count open squares beyond the primed run — future priming potential."""
        dx, dy = self.DIRS[direction]
        x, y = pos
        cx, cy = x + dx, y + dy

        # Skip past existing primed run
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
            if board.get_cell((cx, cy)) == Cell.PRIMED:
                cx += dx
                cy += dy
            else:
                break

        # Count open squares beyond
        count = 0
        for _ in range(6):
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
        """How many of the 4 neighbors of pos are primed?"""
        x, y = pos
        count = 0
        for dx, dy in self.DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if board.get_cell((nx, ny)) == Cell.PRIMED:
                    count += 1
        return count

    def _total_open_reach(self, pos, board) -> int:
        """Count total open and primed squares visible in all 4 directions."""
        x, y = pos
        total = 0
        for dx, dy in self.DIRS.values():
            cx, cy = x + dx, y + dy
            for _ in range(7):
                if not (0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE):
                    break
                cell = board.get_cell((cx, cy))
                if cell in (Cell.SPACE, Cell.PRIMED):
                    total += 1
                    cx += dx
                    cy += dy
                else:
                    break
        return total