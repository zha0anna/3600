from collections.abc import Callable
from typing import List, Tuple
import random

from game import board, move, enums
from game.enums import MoveType, Direction, CARPET_POINTS_TABLE, Cell, BOARD_SIZE


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0
        self.total_points = 0

    def commentate(self):
        return f"Turns: {self.turn} | Points: {self.total_points}"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        self.total_points = board.player_worker.get_points()

        moves = board.get_valid_moves()
        if not moves:
            return None

        best_move, best_score = None, float('-inf')

        for m in moves:
            s = self._score_move(m, board)
            if s > best_score:
                best_score = s
                best_move = m

        return best_move

    # -----------------------------------------------------------------------
    # SCORE A SINGLE MOVE — this is the entire brain of the agent
    # -----------------------------------------------------------------------

    def _score_move(self, m, board) -> float:
        pos = board.player_worker.get_location()

        # --- CARPET: immediate points, heavily preferred ---
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS_TABLE.get(m.roll_length, -99)
            if pts < 2:
                return -50   # never carpet a single square (-1 pts)
            # Bonus for longer rolls — disproportionately valuable
            return pts * 10 + m.roll_length * 5

        # --- PRIME: score by how good the line we are building is ---
        if m.move_type == MoveType.PRIME:
            # How many primed squares already exist in this direction?
            existing = self._count_primed_in_direction(pos, m.direction, board)
            # How much open space exists beyond for future priming?
            open_space = self._open_squares_in_direction(pos, m.direction, board)

            # If we prime here and there are already N primed squares ahead,
            # we could carpet N+1 squares next turn
            future_carpet_len = existing + 1
            future_pts = CARPET_POINTS_TABLE.get(min(future_carpet_len, 7), 21)

            return future_pts * 8 + open_space * 2 + 5  # +5 base: priming is always worthwhile

        # --- PLAIN: use only to reposition toward productive areas ---
        if m.move_type == MoveType.PLAIN:
            dest = self._dest_after_move(pos, m.direction)
            # Score by how many primed squares are adjacent to destination
            adj_primed = self._adjacent_primed_count(dest, board)
            # And how much open space is reachable from destination
            open_reach = self._total_open_reach(dest, board)
            return adj_primed * 6 + open_reach * 0.5 - 5  # -5: plain moves are a last resort

        # --- SEARCH: handled by sensor_data from partner ---
        return -100

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
        Count contiguous primed squares starting from the square
        immediately ahead of pos in direction.
        This is the run we could carpet if we were at the end of it.
        """
        dx, dy = self.DIRS[direction]
        x, y = pos
        count = 0
        cx, cy = x + dx, y + dy  # start at the adjacent square

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
        """
        Count open (SPACE) squares beyond the primed run in this direction.
        Represents future priming potential after the current run.
        """
        dx, dy = self.DIRS[direction]
        x, y = pos
        count = 0

        # Skip past the current primed run first
        cx, cy = x + dx, y + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
            if board.get_cell((cx, cy)) == Cell.PRIMED:
                cx += dx
                cy += dy
            else:
                break

        # Now count open squares beyond the run
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
        """
        Count total open/primed squares visible in all 4 directions from pos.
        Gives a sense of how productive this position is to move to.
        """
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