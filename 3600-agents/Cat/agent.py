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
        self.turn_count = 0

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
            # Apply our own last search result (miss → zero out; catch → respawn)
            my_loc, my_result = board.player_search
            if my_loc is not None:
                if my_result:
                    self.belief_tracker.reset_to_spawn()
                else:
                    self.belief_tracker.apply_search_evidence(my_loc, False)

            # Player A's very first turn: rat moved once from T_1000, so only 1 T step needed.
            # Every other turn (including Player B's turn 1): rat moved twice, so 2 T steps needed.
            is_first_a = (self.turn_count == 0 and board.is_player_a_turn)
            if not is_first_a:
                self.belief_tracker.predict()
                opp_loc, opp_result = board.opponent_search
                self.belief_tracker.apply_search_evidence(opp_loc, opp_result)

            self.belief_tracker.update(board, sensor_data)
            self.turn_count += 1
            self.last_estimate, self.last_confidence, _ = self.belief_tracker.estimate()

        moves = board.get_valid_moves()

        if self.belief_tracker is not None and self.last_confidence >= 0.16:
            return move.Move.search(self.last_estimate)

        return random.choice(moves)
    
    def calculate_search_ev(self, confidence: float, turn_count: int) -> float:
        """
        Calculates the Expected Value of guessing the rat's location.
        """
        # 1. The Base Point Expectation
        # Hit = +4 points, Miss = -2 points
        prob_hit = confidence
        prob_miss = 1.0 - confidence
        
        base_ev = (prob_hit * 4.0) + (prob_miss * -2.0)
        
        # 2. The Reset Benefit (Heuristic)
        # Catching the rat forces it to respawn and run silently for 1000 steps.
        # This destroys any tracking progress your opponent has made.
        # This is more valuable early/mid game when the opponent has time to capitalize on a catch.
        turns_remaining = 40 - (turn_count // 2) 
        reset_value = 0.5 * (turns_remaining / 40.0) 
        expected_reset_bonus = prob_hit * reset_value
        
        # 3. The Information Leakage Cost (Heuristic)
        # Missing gives your opponent a guaranteed 0.0 probability for that square.
        # This is essentially free intel that refines their HMM without them spending a turn.
        # It's most costly when your confidence was low (meaning you gave away a highly likely square).
        intel_cost = 0.2 
        expected_intel_penalty = prob_miss * intel_cost
        
        # Total True EV of the action
        total_ev = base_ev + expected_reset_bonus - expected_intel_penalty
        
        return total_ev