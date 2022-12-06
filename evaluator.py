

from collections import defaultdict
from copy import deepcopy
import heapq
from random import choices
from typing import Generator, Optional, Tuple
import chess
import torch
import torch.nn as nn
import numpy as np
import reconchess
from net import BeliefNet

from state import ID_MAPPING

from state import BeliefState
from utils.lc0 import LeelaWrapper

class BoardSample:
    def __init__(self, indices, layers) -> None:
        self.layer_counts = layers
        self.selected_indices = indices



class Evaluator:
    def __init__(self, leela) -> None:
        #self.state = GameState()
        #self.op_state = GameState()
        self.model = BeliefNet(22,8)
        self.engine = LeelaWrapper() if leela is None else leela
        # should opp be a seperate (identical) evaluator, or something else?
        # how do we simulate opponets moves (given an uncertain inital board state)?
        pass
    
    def choose_sense_square(self, state: BeliefState) -> int:
        # convert state to BeliefNet input
        nn_input = state.to_nn_input() # TODO: move to target device
        
         # get BeliefNet output and update the belief state with the result
        with torch.no_grad():
            result: torch.Tensor = self.model(torch.from_numpy(np.expand_dims(nn_input, 0)), pi=state.num_opp_pieces)
            state.update(*[r.squeeze(0).numpy() for r in result])

        # accumulate weighted eval scores for each piece on each grid space
        categorical_scores = defaultdict(lambda: defaultdict(lambda: {'running_score': 0.0, 'running_weight_sum': 0.0}))

        for board, prob in self.get_at_most_n_likely_states(state, n=100):
            
            # check to see if opponet is in check, if so we weight this board with maximum weight
            if board.is_check():
                weighted_evaluation = 100000
            else:
                weighted_evaluation = self.engine.get_engine_centipawn_eval(board) * prob
            # for each grid space
            for s in range(64):
                piece = board.piece_at(s)
                categorical_scores[s][piece]['running_score'] += weighted_evaluation
                categorical_scores[s][piece]['running_weight_sum'] += prob
        
        
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
    
    def choose_move(self, state: BeliefState) -> Optional[reconchess.chess.Move]:
        # convert belief state to BeliefNet input
        nn_input = state.to_nn_input() # TODO: move to target device
        # get BeliefNet output and update the belief state with the result
        with torch.no_grad():
            result: torch.Tensor = self.model(torch.from_numpy(np.expand_dims(nn_input, 0)), pi=state.num_opp_pieces)
            state.update(*[r.squeeze(0).numpy() for r in result])

        # accumulate LC0 probabilities for each move from each of N most likely states,
        # weighted by board likelihood
        move_scores = dict()
        for board, prob in self.get_at_most_n_likely_states(state, n=100):
            if board.is_check():
                probs = dict()
                # get square of enemy king
                king_sq = board.king(chess.BLACK)
                for sq in board.checkers():
                    if board.piece_at(sq).color == chess.WHITE:
                        move = reconchess.chess.Move(from_square=sq, to_square=king_sq)
                        probs[move.uci()] = 1
            else:
                probs = self.engine.get_move_probabilities(board)

            for m, p in probs.items():
                if m not in move_scores:
                    move_scores[m] = 0.0
                move_scores[m] += (prob * p)

        # choose the best scoring move
        if move_scores:
            return reconchess.chess.Move.from_uci(max(move_scores, key=move_scores.get))
        else:
            print('No moves to choose from!')
            return None


    @staticmethod
    def np_to_board(state: BeliefState, locs: np.ndarray, en_passant_file: Optional[int], op_castle_q: bool, op_castle_k: bool) -> reconchess.chess.Board:
        board = deepcopy(state.board)
        board.turn = chess.WHITE
        board.ep_square = chess.Square(en_passant_file + 40) if en_passant_file is not None else None
        for p, r, c in locs:
            sq = (r * 8) + c
            if p == 0:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.KING, color=chess.BLACK))
            elif p == 1:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.QUEEN, color=chess.BLACK))
            elif p == 2:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.ROOK, color=chess.BLACK))
            elif p == 3:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.KNIGHT, color=chess.BLACK))
            elif p == 4:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.BISHOP, color=chess.BLACK))
            elif p == 5:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.PAWN, color=chess.BLACK))
        fen_castle_string = ''
        if state.board.has_kingside_castling_rights(chess.WHITE):
            fen_castle_string += 'K'
        if state.board.has_queenside_castling_rights(chess.WHITE):
            fen_castle_string += 'Q'
        if op_castle_k:
            fen_castle_string += 'k'
        if op_castle_q:
            fen_castle_string += 'q'
        board.set_castling_fen(fen_castle_string)
        return board



    def get_at_most_n_likely_states(self, state: BeliefState, n=100):
        state.zero_occupied_spaces()
        state.zero_backrank_pawns()
        grid_spaces = []
        
        seen = set()

        # for each hypothesis, we sample:
        # 16 next most likely piece locations (or less if pieces have been captured)
        # King + Queen -side castling rights
        # En passant file (or no file)
        
        
        num_samples = state.num_opp_pieces
        count = 0
        # yields a sorted list of lists (by probability)
        # each list contains the probability and coordinates for each piece at a particular square
        for r in range(8):
            for c in range(8):
                grid_spaces.append(sorted([(state.opp_board[p][r][c], (p, r, c)) for p in range(6)], key= lambda x: -x[0]))
        grid_spaces.sort(reverse=True)
        # each object that goes on the heap is a tuple containing the probability product and (queue index, square choice index)
        # we also need to consider the most likely state for each of queenside castling, kingside castling, and en passant square and include those in the probability product
    
        en_passant_files = sorted([(x, i) for i,x in enumerate(state.opp_en_passant)] + [(np.prod([1 - x for x in state.opp_en_passant]), -1)], key=lambda x: -x[0])
        kingside_castle = sorted([(state.opp_castle_k, True), (1 - state.opp_castle_k, False)], key=lambda x: -x[0])
        queenside_castle = sorted([(state.opp_castle_q, True), (1 - state.opp_castle_q, False)], key=lambda x: -x[0])

        first = (-np.prod([s for s, _ in grid_spaces[0][0:num_samples]]) * en_passant_files[0][0] * kingside_castle[0][0] * queenside_castle[0][0], (tuple([(i,0) for i in range(num_samples)]), 0, 0, 0))
        heap = [first]

        while heap and count < n:
            # get most likely combo from heap
            prob, configuration = heapq.heappop(heap)
            prob = -prob

            if configuration not in seen and prob != 0.0:
                # we need to keep track of boards we've already seen (downside of this algorithm, it can yield duplicates)
                seen.add(configuration)
                selections, en_passant, castle_k, castle_q = configuration
                board = self.np_to_board(state, [grid_spaces[i][j][1] for i, j in selections], en_passant_files[en_passant][1], queenside_castle[castle_q][1], kingside_castle[castle_k][1])
            
                status = board.status()

                if not status or not (\
                        (chess.Status.TOO_MANY_BLACK_PAWNS | status) == status or
                        (chess.Status.TOO_MANY_KINGS | status) == status or 
                        (chess.Status.NO_BLACK_KING | status) == status):
                    count += 1
                    yield (board, prob)
                # select next spaces

                # pivot king to next most likely square AND
                # pivot other squares

                
                for i in range(num_samples):
                    # cannot relax if next selection occurs right after

                    # 1. we can relax grid space
                    if (i < num_samples - 1) and (selections[i][0] < len(grid_spaces) - 1) and (selections[i][0] + 1 != selections[i + 1][0]):
                        new_selections = list(selections)
                        # update index i with new relaxed selection
                        new_selections[i] = (selections[i][0] + 1, selections[i][1])

                        old_val, new_val = grid_spaces[selections[i][0]][selections[i][1]][0], grid_spaces[selections[i][0] + 1][selections[i][1]][0]
                        new_prob = prob / old_val * new_val
                        new_selections = tuple(new_selections)
                        heapq.heappush(heap, (-new_prob, (new_selections, en_passant, castle_k, castle_q)))

                    # we can relax piece chosen
                    if selections[i][1] < 6 - 1:
                        new_selections = list(selections)
                        # update index i with new relaxed selection
                        new_selections[i] = (selections[i][0], selections[i][1] + 1)

                        old_val, new_val = grid_spaces[selections[i][0]][selections[i][1]][0], grid_spaces[selections[i][0]][selections[i][1] + 1][0]

                        new_prob = prob / old_val * new_val
                        new_selections = tuple(new_selections)
                        heapq.heappush(heap, (-new_prob, (new_selections, en_passant, castle_k, castle_q)))
                
                # relax en passant square
                if en_passant < 8:
                    en_passant += 1
                    new_prob = prob / en_passant_files[en_passant - 1][0] * en_passant_files[en_passant][0]
                # relax kingside castling
                if castle_k == 0:
                    castle_k = 1
                    new_prob = prob / kingside_castle[0][0] * kingside_castle[castle_k][0]
                    heapq.heappush(heap, (-new_prob, (selections, en_passant, castle_k, castle_q)))
                # relax queenside castling
                if castle_q == 0:
                    castle_q = 1
                    new_prob = prob / queenside_castle[0][0] * queenside_castle[castle_q][0]
                    heapq.heappush(heap, (-new_prob, (selections, en_passant, castle_k, castle_q)))

        return


        
                
                
            
        
        



