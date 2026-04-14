from collections.abc import Callable
from typing import List, Set, Tuple
import random

from game import board, move, enums
from game.enums import MoveType, Direction, CARPET_POINTS_TABLE


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        self.turn = 0
        self.total_points = 0
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return f"Completed {self.turn} turns with a total of {self.total_points} points."

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn += 1
        self.total_points = board.player_worker.get_points()
        moves = board.get_valid_moves()
        if not moves:
            return None

        #moves by type
        carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET]
        prime_moves  = [m for m in moves if m.move_type == MoveType.PRIME]
        plain_moves  = [m for m in moves if m.move_type == MoveType.PLAIN]

        if carpet_moves: #prioritize highest scoring carpet move
            best = carpet_moves[0]
            best_score = CARPET_POINTS_TABLE.get(best.roll_length)
            for m in carpet_moves: #find best scoring carpet move
                score = CARPET_POINTS_TABLE.get(m.roll_length)
                if score > best_score:
                    best = m
                    best_score = score

            if CARPET_POINTS_TABLE.get(best.roll_length) >= 2:  # only carpet if it scores positive
                return best

        if prime_moves:
            best_prime = self.best_prime(prime_moves, board)
            return best_prime

        if plain_moves:
            return random.choice(plain_moves)

        return random.choice(moves)

    #scores prime moves, prefers those with primed/open squares in direction of prime
    def best_prime(self, prime_moves, board):
        best_move = prime_moves[0]
        best_score = -1

        pos = board.player_worker.get_location()

        for m in prime_moves:
            score = self.open_squares_in_direction(pos, m.direction, board)
            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    #counts open and primed squares in the direction of a prime move, up to 6 squares away (max carpet roll minus the primed square itself)
    def open_squares_in_direction(self, pos, direction, board):
        x, y = pos
        dir_dict = {
            Direction.UP:    (0, -1),
            Direction.DOWN:  (0,  1),
            Direction.LEFT:  (-1, 0),
            Direction.RIGHT: ( 1, 0),
        }
        dx, dy = dir_dict[direction]

        count = 0 #score
        cx, cy = x + dx, y + dy 
        cx, cy = cx + dx, cy + dy  # go 2 steps in direction (to skip primed square itself)

        for i in range(6):  # max carpet roll is 7, minus the primed square itself
            if not (0 <= cx < 8 and 0 <= cy < 8): # out of bounds
                break
            cell = board.get_cell((cx, cy))
            if cell == enums.Cell.PRIMED:
                count += 2
            elif cell == enums.Cell.SPACE:
                count += 1 
            else: #blocked or carpeted
                break
            cx += dx #next step
            cy += dy

        return count

        