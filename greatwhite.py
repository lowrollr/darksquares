

from typing import List, Optional, Tuple
from numpy import take
import reconchess
import random

from evaluator import Evaluator
from net import BeliefNet

from state import BeliefState

class GreatWhiteBot(reconchess.Player):
    def __init__(self, evaluator) -> None:
        super().__init__()
        self.color = None
        self.beliefs: BeliefState = None
        self.evaluator: Evaluator = evaluator
        self.__name__ = 'GreatWhiteBot'

    def handle_game_start(self, color: reconchess.Color, board: reconchess.chess.Board, opponent_name: str):
        self.color = color
        self.beliefs = BeliefState(color)


    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[reconchess.Square]):
        self.beliefs.opp_move(capture_square)

    def choose_sense(self, sense_actions: List[reconchess.Square], move_actions: List[reconchess.chess.Move], seconds_left: float) -> \
            Optional[reconchess.Square]:
        if self.beliefs.board.move_stack:
            return self.evaluator.choose_sense_square(self.beliefs)
        else:
            return 0

    def handle_sense_result(self, sense_result: List[Tuple[reconchess.Square, Optional[reconchess.chess.Piece]]]):
        print(sense_result)
        self.beliefs.set_ground_truth(sense_result)


    def choose_move(self, move_actions: List[reconchess.chess.Move], seconds_left: float) -> Optional[reconchess.chess.Move]:
        return self.evaluator.choose_move(self.beliefs)

    def handle_move_result(self, requested_move: Optional[reconchess.chess.Move], taken_move: Optional[reconchess.chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[reconchess.Square]):
        

        if taken_move != requested_move:
            self.beliefs.apply_impl(requested_move, taken_move)
        if capture_square:
            self.beliefs.capture(capture_square)
        if taken_move:
            self.beliefs.apply_move(taken_move)
        else:
            self.beliefs.board.push(reconchess.chess.Move.from_uci('0000'))
        self.beliefs.board.push(reconchess.chess.Move.from_uci('0000'))

        print(f'Requested: {requested_move}, Taken: {taken_move}')
        print(self.beliefs.board)

    def handle_game_end(self, winner_color: Optional[reconchess.Color], win_reason: Optional[reconchess.WinReason],
                        game_history: reconchess.GameHistory):
        print(game_history._taken_moves, win_reason)
        self.evaluator.engine.quit()