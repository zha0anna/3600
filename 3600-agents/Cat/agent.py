from collections.abc import Callable
from typing import List, Set, Tuple
import random

from game import board, move, enums
from rat_belief import RatBeliefHMM


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
        self.belief_tracker = None
        self.last_estimate = None
        self.last_confidence = 0.0

        if transition_matrix is not None:
            self.belief_tracker = RatBeliefHMM(transition_matrix)
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        if self.belief_tracker is not None:
            search_loc, search_result = board.opponent_search
            self.belief_tracker.apply_search_evidence(search_loc, search_result)
            self.belief_tracker.update(board, sensor_data)
            self.last_estimate, self.last_confidence, _ = self.belief_tracker.estimate()

        moves = board.get_valid_moves()

        if self.belief_tracker is not None and self.last_confidence >= 0.16:
            return move.Move.search(self.last_estimate)

        return random.choice(moves)
