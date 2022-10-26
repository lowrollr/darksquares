


from evaluator import Evaluator
import numpy as np
from collections import defaultdict
import reconchess

from state import BeliefState

class TrainingEvaluator(Evaluator):
    def __init__(self, leela) -> None:
        super().__init__(leela)
    
    def get_input(self, state: BeliefState) -> np.ndarray:
        return state.to_nn_input()

    def get_sense_square(self, sense_output, state: BeliefState):
        state.update(*[r for r in sense_output])

        categorical_scores = defaultdict(lambda: defaultdict(lambda: {'running_score': 0.0, 'running_weight_sum': 0.0}))

        for np_board, prob_board in self.get_at_most_n_likely_states(state, n=100):
            board: reconchess.chess.Board = self.np_to_board(state, np_board)
            # check to see if opponet is in check, if so we weight this board with maximum weight
            if board.is_check():
                weighted_evaluation = 100000
            else:
                weighted_evaluation = self.engine.get_engine_centipawn_eval(board) * prob_board
            # for each grid space
            for s in range(64):
                piece = board.piece_at(s)
                categorical_scores[s][piece]['running_score'] += weighted_evaluation
                categorical_scores[s][piece]['running_weight_sum'] += prob_board
        
        
        # calculate eval variance for each square on the board
        square_variances = np.ndarray(shape=(8,8))

        for sq in categorical_scores:
            avg_scores = []
            for piece in categorical_scores[sq]:
                weighted_avg = categorical_scores[sq][piece]['running_score']/categorical_scores[sq][piece]['running_weight_sum']
                avg_scores.append(weighted_avg)
            sq_variance = np.var(avg_scores)
            r, c = sq // 8, sq % 8
            square_variances[r][c] = sq_variance
        
        rolling_variances = np.sum(np.lib.stride_tricks.sliding_window_view(square_variances, (3,3)), axis=(3,2))
        
        # choose the 3x3 square that maximizes measured variance and return it
        best_sq = np.argmax(rolling_variances) + 9
        return best_sq

    def get_best_move(self, belief_output, state: BeliefState):
        state.update(* [r for r in belief_output])
        # accumulate LC0 probabilities for each move from each of N most likely states,
        # weighted by board likelihood
        move_scores = dict()
        for np_board, prob_board in self.get_at_most_n_likely_states(state, n=100):
            board = self.np_to_board(state, np_board)
            if board.is_check():
                probs = dict()
                # get square of enemy king
                king_sq = np.argmax(np_board[0], axis=0)
                for sq in board.checkers():
                    if board.piece_at(sq).color == state.white:
                        move = reconchess.chess.Move(from_square=sq, to_square=king_sq)
                        probs[move.uci()] = 1
            else:
                probs = self.engine.get_move_probabilities(board)

            for m, p in probs.items():
                if m not in move_scores:
                    move_scores[m] = 0.0
                move_scores[m] += (prob_board * p)

        # choose the best scoring move
        if move_scores:
            return reconchess.chess.Move.from_uci(max(move_scores, key=move_scores.get))
        else:
            print('No moves to choose from!')
            return None