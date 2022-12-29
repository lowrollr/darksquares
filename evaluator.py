

from collections import defaultdict
from copy import deepcopy
import heapq
from random import choices
import time
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

def np_to_board(state: BeliefState, locs, en_passant_file: int, op_castle_q: bool, op_castle_k: bool) -> reconchess.chess.Board:
    board = deepcopy(state.board)
    board.turn = chess.WHITE
    board.ep_square = chess.Square(en_passant_file + 40) if en_passant_file != -1 else None
    for _, (p, r, c) in locs:
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

class BoardSample:
    def __init__(self, indices, layers) -> None:
        self.layer_counts = layers
        self.selected_indices = indices

class Evaluator:
    def __init__(self, leela) -> None:
        #self.state = GameState()
        #self.op_state = GameState()
        self.model = BeliefNet(22,8)
        checkpoint = torch.load('model.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.engine = LeelaWrapper() if leela is None else leela
        # should opp be a seperate (identical) evaluator, or something else?
        # how do we simulate opponets moves (given an uncertain inital board state)?
        pass
    
    def choose_sense_square(self, state: BeliefState) -> int:
        # convert state to BeliefNet input
        nn_input = state.to_nn_input() # TODO: move to target device
        
         # get BeliefNet output and update the belief state with the result
        with torch.no_grad():
            result: torch.Tensor = self.model(torch.from_numpy(np.expand_dims(nn_input, 0)))
            state.update(*[r.squeeze(0).numpy() for r in result])

        # accumulate weighted eval scores for each piece on each grid space
        categorical_scores = defaultdict(lambda: defaultdict(lambda: {'running_score': 0.0, 'running_weight_sum': 0.0}))

        for board, prob in self.get_at_most_n_likely_states(state, n=100):
            
            # check to see if opponent is in check, if so we weight this board with maximum weight
            if board.is_checkmate():
                continue
            board.turn = reconchess.chess.BLACK
            if board.is_check():
                weighted_evaluation = 100000
            else:
                board.turn = reconchess.chess.WHITE
                weighted_evaluation = self.engine.get_engine_centipawn_eval(board) * prob
                # weighted_evaluation = prob
            board.turn = reconchess.chess.WHITE
            # for each grid space
            for s in range(64):
                piece = board.piece_at(s)
                if piece is None or piece.color == reconchess.chess.BLACK:
                    categorical_scores[s][piece]['running_score'] += weighted_evaluation
                    categorical_scores[s][piece]['running_weight_sum'] += prob
        
        
        # calculate eval variance for each square on the board
        square_variances = np.ndarray(shape=(8,8))

        for sq in categorical_scores:
            avg_scores = []
            for piece in categorical_scores[sq]:
                weighted_avg = categorical_scores[sq][piece]['running_score']/categorical_scores[sq][piece]['running_weight_sum']
                avg_scores.append(weighted_avg)
            
            r, c = sq // 8, sq % 8
            if avg_scores:
                square_variances[r][c] = np.var(avg_scores)
            else:
                square_variances[r][c] = 0.0
        
        rolling_variances = np.sum(np.lib.stride_tricks.sliding_window_view(square_variances, (3,3)), axis=(3,2))
        
        # choose the 3x3 square that maximizes measured variance and return it
        best_sq = np.argmax(rolling_variances)
        # center square from 6 * 6 grid onto 8 * 8 grid
        best_sq = (best_sq // 6) * 8 + (best_sq % 6) + 9
        return state.get_square(best_sq)
    
    def choose_move(self, state: BeliefState) -> Optional[reconchess.chess.Move]:
        # convert belief state to BeliefNet input
        nn_input = state.to_nn_input() # TODO: move to target device
        # get BeliefNet output and update the belief state with the result
        with torch.no_grad():
            result: torch.Tensor = self.model(torch.from_numpy(np.expand_dims(nn_input, 0)))
            state.update(*[r.squeeze(0).numpy() for r in result])

        # accumulate LC0 probabilities for each move from each of N most likely states,
        # weighted by board likelihood
        move_scores = dict()
        for board, prob in self.get_at_most_n_likely_states(state, n=100):
            if board.is_checkmate():
                continue
            board.turn = reconchess.chess.BLACK
            if board.is_check():
                probs = dict()
                # get square of enemy king
                king_sq = board.king(reconchess.chess.BLACK)
                for sq in board.checkers():
                    if board.piece_at(sq).color == reconchess.chess.WHITE:
                        move = reconchess.chess.Move(from_square=sq, to_square=king_sq)
                        probs[move.uci()] = 1
            else:
                board.turn = reconchess.chess.WHITE
                probs = self.engine.get_move_probabilities(board)
            board.turn = reconchess.chess.WHITE
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
    def get_at_most_n_likely_states(state: BeliefState, n=100):
        state.zero_occupied_spaces()
        state.zero_backrank_pawns()
        
        samples = set()
        boards = set()
        
        num_samples = state.num_opp_pieces
        count = 0

        king_positions = sorted([(state.opp_board[0][r][c], (0, r, c)) for r in range(8) for c in range(8)], key=lambda x: -x[0])
        pawn_positions = sorted([(state.opp_board[5][r][c], (5, r, c)) for r in range(8) for c in range(8)], key=lambda x: -x[0])
        piece_positions = sorted([sorted([(state.opp_board[p][r][c], (p, r, c)) for p in range(1, 5)], key=lambda x: -x[0]) for r in range(8) for c in range(8)], key=lambda x: -x[0][0])
        en_passant_files = sorted([(x, i) for i,x in enumerate(state.opp_en_passant)] + [(np.prod([1 - x for x in state.opp_en_passant]), -1)], key=lambda x: -x[0])
        kingside_castle = sorted([(state.opp_castle_k, True), (1 - state.opp_castle_k, False)], key=lambda x: -x[0])
        queenside_castle = sorted([(state.opp_castle_q, True), (1 - state.opp_castle_q, False)], key=lambda x: -x[0])

        def get_hypothesis_board(pieces, pawns, king, en_passant, castle_k, castle_q):
            most_likely_locs = tuple(sorted([(piece_positions[i][p]) for (i,p) in pieces] + [pawn_positions[i] for i in pawns], key=lambda x: -x[0])[:num_samples-1])            
            return np.prod([x[0] for x in most_likely_locs]) * king_positions[king][0] * en_passant_files[en_passant][0] * kingside_castle[castle_k][0] * queenside_castle[castle_q][0], most_likely_locs + ((0, king_positions[king][1]),)

        # a board hypothesis is composed of
        # 1. position of enemy king
        # 2. position of other enemy pieces
        # 3. en passant location
        # 4. castling rights (kingside and queenside)

        # get initial valid hypothesis

        # get piece positions (ensure # of pawns is not > 8)

        first = (tuple([(i,0) for i in range(num_samples-1)]), tuple([i for i in range(8)]), 0, 0, 0, 0)
        prob, likely_locs = get_hypothesis_board(*first)

        heap = [(-prob, first, likely_locs)]

        def heap_hypothesis(prob, hypothesis, locs):
            if locs in samples:
                return
            samples.add(locs)
            heapq.heappush(heap, (-prob, hypothesis, locs))

        while heap and count < n:
            # get most likely combo from heap
            prob, hypothesis, locs = heapq.heappop(heap)
            

            pieces, pawns, king, en_passant, castle_k, castle_q = hypothesis

            # get board from hypothesis
            board = np_to_board(state, locs, en_passant_files[en_passant][1], queenside_castle[castle_q][1], kingside_castle[castle_k][1])
            
            fen = board.fen()
            if fen not in boards:
                count += 1
                boards.add(fen)
                yield (board, -prob)

            # relax hypothesiss to generate next hypotheses

            # 1. we can relax king position
            if king < len(king_positions) - 1:
                new_hypothesis = (pieces, pawns, king + 1, en_passant, castle_k, castle_q)
                new_prob, locs = get_hypothesis_board(*new_hypothesis)
                heap_hypothesis(new_prob, new_hypothesis, locs)

            # 2. we can relax en passant
            if en_passant < len(en_passant_files) - 1:
                new_hypothesis = (pieces, pawns, king, en_passant + 1, castle_k, castle_q)
                new_prob, locs = get_hypothesis_board(*new_hypothesis)
                heap_hypothesis(new_prob, new_hypothesis, locs)

            # 3. we can relax castling rights
            if castle_k < len(kingside_castle) - 1:
                new_hypothesis = (pieces, pawns, king, en_passant, castle_k + 1, castle_q)
                new_prob, locs = get_hypothesis_board(*new_hypothesis)
                heap_hypothesis(new_prob, new_hypothesis, locs)

            if castle_q < len(queenside_castle) - 1:
                new_hypothesis = (pieces, pawns, king, en_passant, castle_k, castle_q + 1)
                new_prob, locs = get_hypothesis_board(*new_hypothesis)
                heap_hypothesis(new_prob, new_hypothesis, locs)

            # relax pawns
            for i in range(len(pawns)):
                if (i == len(pawns) - 1 and pawns[i] < len(pawn_positions) - 1) or (i != len(pawns) - 1 and pawns[i] + 1 != pawns[i+1]):
                    new_hypothesis = (pieces, pawns[:i] + (pawns[i] + 1,) + pawns[i+1:], king, en_passant, castle_k, castle_q)
                    new_prob, locs = get_hypothesis_board(*new_hypothesis)
                    heap_hypothesis(new_prob, new_hypothesis, locs)
                

            # relax other pieces
            for i in range(len(pieces)):
                if (i == len(pieces) - 1 and pieces[i][0] < len(piece_positions) - 1) or (i != len(pieces) - 1 and pieces[i][0] + 1 != pieces[i+1][1]):
                    new_hypothesis = (pieces[:i] + ((pieces[i][0] + 1, pieces[i][1]),) + pieces[i+1:], pawns, king, en_passant, castle_k, castle_q)
                    new_prob, locs = get_hypothesis_board(*new_hypothesis)
                    heap_hypothesis(new_prob, new_hypothesis, locs)
                
                if i < 4:
                    new_hypothesis = (pieces[:i] + ((pieces[i][0], pieces[i][1]+1),) + pieces[i+1:], pawns, king, en_passant, castle_k, castle_q)
                    new_prob, locs = get_hypothesis_board(*new_hypothesis)
                    heap_hypothesis(new_prob, new_hypothesis, locs)
        
        return


        
                
                
            
        
        



