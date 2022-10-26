

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

        for np_board, prob_board in self.get_at_most_n_likely_states(state, n=100):
            board: chess.Board = self.np_to_board(state, np_board)
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


    @staticmethod
    def np_to_board(state: BeliefState, locs: np.ndarray) -> reconchess.chess.Board:
        board = deepcopy(state.board)
        for p, r, c in locs:
            sq = (r * 8) + c
            if p == 0:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.KING, color=not state.white))
            elif p == 1:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.QUEEN, color=not state.white))
            elif p == 2:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.ROOK, color=not state.white))
            elif p == 3:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.KNIGHT, color=not state.white))
            elif p == 4:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.BISHOP, color=not state.white))
            elif p == 5:
                board.set_piece_at(sq, reconchess.chess.Piece(reconchess.chess.PAWN, color=not state.white))
        return board

    def get_at_most_n_likely_states(self, state: BeliefState, n=100):
        
        grid_spaces = []
        
        seen = set()
        
        num_samples = state.num_opp_pieces
        count = 0
        num_pieces = 6
        # yields a sorted list of lists (by probability)
        # each list contains the probability and coordinates for each piece at a particular square
        for r in range(8):
            for c in range(8):
                grid_spaces.append(sorted([(state.opp_beliefs[p][r][c], (p, r, c)) for p in range(6)], key= lambda x: -x[0]))
        grid_spaces.sort(reverse=True)
        # each object that goes on the heap is a tuple containing the probability product and (queue index, square choice index)
        first = (-np.prod([s for s, _ in grid_spaces[0][0:num_samples]]), tuple([(i,0) for i in range(num_samples)]))
        heap = [first]

        while heap and count < n:
            # get most likely combo from heap
            prob, selections = heapq.heappop(heap)
            prob = -prob

            if selections not in seen:
                seen.add(selections)
                # add coordinates of each piece in combo to solution
                # make sure number of kings on board == 1:
                pseudo_legal = True
                has_king = False
                for i, j in selections:
                    p,r,c = grid_spaces[i][j][1]
                    if p == 0:
                        if has_king:
                            pseudo_legal = False
                            break
                        has_king = True
                    elif p == 5:
                        if r == 0 or r == 7:
                            pseudo_legal = False
                            break

                pseudo_legal = pseudo_legal and has_king
                if pseudo_legal:
                    count += 1
                    yield ([[grid_spaces[i][j][1] for i, j in selections], prob])
                # select next spaces
                
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
                        heapq.heappush(heap, (-new_prob, new_selections))

                    # we can relax piece chosen
                    if selections[i][1] < num_pieces - 1:
                        new_selections = list(selections)
                        # update index i with new relaxed selection
                        new_selections[i] = (selections[i][0], selections[i][1] + 1)

                        old_val, new_val = grid_spaces[selections[i][0]][selections[i][1]][0], grid_spaces[selections[i][0]][selections[i][1] + 1][0]

                        new_prob = prob / old_val * new_val
                        new_selections = tuple(new_selections)
                        heapq.heappush(heap, (-new_prob, new_selections))
                
        return


        
                
                
            
        
        



