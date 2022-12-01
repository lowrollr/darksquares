
from copy import deepcopy
from typing import List, Optional, Tuple
import chess

import numpy as np
import reconchess
import torch
from matplotlib.pyplot import pie

from utils.board import (convert_squares_to_coords, get_squares_between_incl,
                         opposite_square)

ID_MAPPING = {
    reconchess.chess.KING:   0,
    reconchess.chess.QUEEN:  1,
    reconchess.chess.ROOK:   2,
    reconchess.chess.KNIGHT: 3,
    reconchess.chess.BISHOP: 4,
    reconchess.chess.PAWN:   5
}




        

class BeliefState:
    def __init__(self, playing_white) -> None:
        self.board = reconchess.chess.Board('8/8/8/8/8/8/PPPPPPPP/RNBQKBNR')
        self.psuedo_absences = np.zeros(shape=(8,8))
        self.psuedo_presences = np.zeros(shape=(8,8))
        self.known_absences = np.zeros(shape=(8,8))
        self.known_presences = np.zeros(shape=(8,8))
        self.num_moves = 0
        self.num_opp_pieces = 16
        self.opp_board = np.zeros(shape=(6,8,8))
        self.opp_en_passant = np.zeros(shape=(1,8))
        self.opp_castle_q = 1
        self.opp_castle_k = 1
        self.white = playing_white
        if not self.white:
            self.board.turn == chess.BLACK
        self.init_opp_board()

        # NOTE: any data must be normalized prior to being used in the state model, 
        # right now each externally called function applies normalization before doing anything else, 
        # perhaps there is a better way to do this
    
        
    def init_opp_board(self):
        for piece, i in ID_MAPPING.items():
            if piece == reconchess.chess.PAWN:
                self.opp_board[i][6] = 1
            elif piece == reconchess.chess.KNIGHT:
                self.opp_board[i][7][1] = 1
                self.opp_board[i][7][6] = 1
            elif piece == reconchess.chess.BISHOP:
                self.opp_board[i][7][2] = 1
                self.opp_board[i][7][5] = 1
            elif piece == reconchess.chess.ROOK:
                self.opp_board[i][7][0] = 1
                self.opp_board[i][7][7] = 1
            elif piece == reconchess.chess.QUEEN:
                self.opp_board[i][7][3] = 1
            elif piece == reconchess.chess.KING:
                self.opp_board[i][7][4] = 1

    def get_square(self, sq) -> reconchess.Square:
        if self.white:
            return sq
        else:
            return opposite_square(sq)
    
    def clear_information_gain(self) -> None:
        self.psuedo_absences = np.zeros(shape=(8,8))
        self.psuedo_presences = np.zeros(shape=(8,8))
        self.known_absences = np.zeros(shape=(8,8))
        self.known_presences = np.zeros(shape=(8,8))

    def zero_occupied_spaces(self) -> None:
        for r in range(8):
            for c in range(8):
                sq = (r * 8) + c
                if self.board.piece_at(sq):
                    self.opp_board[:, r, c] = 0
    

    def to_nn_input(self) -> np.ndarray:
        data_tensor = np.zeros(shape=(22,8,8), dtype=np.float32)
        # -- INFORMATION GAIN -- #
        data_tensor[0] = self.psuedo_absences
        data_tensor[1] = self.known_absences
        data_tensor[2] = self.psuedo_presences
        data_tensor[3] = self.known_presences
        data_tensor[4] = np.unpackbits(np.array([self.num_moves], dtype=np.uint8), count=8)
        data_tensor[5] = np.unpackbits(np.array([self.num_opp_pieces], dtype=np.uint8), count=8)

        # -- CASTLING (US) -- # 
        if self.board.has_kingside_castling_rights(chess.WHITE):
            data_tensor[6,:,0:4] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            data_tensor[6,:,4:] = 1

        # -- EN PASSANT (US) -- # 
        if len(self.board.move_stack) > 1:
            move = self.board.move_stack[-2]
            # so jank bro
            if ''.join(filter(lambda x: not x.isalpha(), move.uci())) == '24':
                f = ord(move.uci()[0]) - 97
                data_tensor[7,:,f] = 1

        # -- PIECES (US) -- #
        for sq, piece in self.board.piece_map().items():
            v = ID_MAPPING[piece.piece_type] + 8
            # convert to normalized square
            r, c = convert_squares_to_coords([sq])[0]
            data_tensor[v][r][c] = 1

        # -- PIECES (THEM) -- #
        data_tensor[14:20] = self.opp_board
        
        # -- EN PASSANT (THEM) -- #
        data_tensor[20] = np.tile(self.opp_en_passant, (8,1))
        
        # -- CASTLING (THEM) -- #
        data_tensor[21:,:,0:4] = self.opp_castle_q
        data_tensor[21:,:,4:] = self.opp_castle_k 

        return data_tensor
        
    

    def opp_move(self, capture_square: Optional[reconchess.Square]) -> None:
        # remove piece that was captured from our local board
        if capture_square is not None:
            sq = self.get_square(capture_square)
            r, c = convert_squares_to_coords([sq])[0]
            self.board.remove_piece_at(capture_square)
            self.known_presences[r][c] = 1

        for r in range(8):
            for c in range(8):
                sq = (r * 8) + c
                if self.board.piece_at(sq):
                    self.known_absences[r][c] = 1

    def update(self, probs, passant, castle):

        self.opp_board = probs
        self.opp_en_passant = passant
        self.opp_castle_q = castle[0]
        self.opp_castle_k = castle[1]
        
    # apply the move to our board, and make sure any gleaned information about our opponet's pieces makes it into our belief state
    def apply_move(self, move: reconchess.chess.Move) -> None:
        from_sq, to_sq = self.get_square(move.from_square), self.get_square(move.to_square)
        move = reconchess.chess.Move(from_sq, to_sq, move.promotion)
        in_between_squares = []
        piece = self.board.piece_at(from_sq)
        # if moving a grounded multi-square piece (Bishop/Rook/Queen), we know all squares in between our starting and ending square were unoccupied by an opposing piece
        # so we can set them to zero and adjust all other probabilities on the board to maintain implied piece count
        if piece.piece_type in {reconchess.chess.ROOK, reconchess.chess.QUEEN, reconchess.chess.BISHOP}:
            in_between_squares = get_squares_between_incl(from_sq, to_sq)
        elif piece.piece_type == reconchess.chess.KING:
            # check if this was a castle move, if so the squares between the rook and the king were unoccupied
            if move.xboard() == 'O-O-O':
                in_between_squares = get_squares_between_incl(from_sq, from_sq - 3)
            elif move.xboard() == 'O-O':
                in_between_squares = get_squares_between_incl(from_sq, from_sq + 2)
            else:
                in_between_squares = [from_sq, to_sq]

        elif piece.piece_type == reconchess.chess.PAWN:
            # check if moved 2 spaces, space between was unoccuppied if so
            if abs(from_sq - to_sq) == 16:
                in_between_squares = get_squares_between_incl(from_sq, to_sq)
            else:
                in_between_squares = [from_sq, to_sq]
        else:
            # knights go where they want to
            in_between_squares = [from_sq, to_sq]

        self.board.push(move)
        for r in range(8):
            for c in range(8):
                sq = (r * 8) + c
                if self.board.piece_at(sq):
                    self.psuedo_absences[r][c] = 1

        for (r,c) in convert_squares_to_coords(in_between_squares):
            self.psuedo_absences[r][c] = 1

    def zero_backrank_pawns(self):
        self.opp_board[ID_MAPPING[reconchess.chess.PAWN]][0][:] = 0
        self.opp_board[ID_MAPPING[reconchess.chess.PAWN]][7][:] = 0

    def set_ground_truth(self, truth: List[Tuple[reconchess.Square, Optional[reconchess.chess.Piece]]]):
        for sq, piece in truth:
            r, c = convert_squares_to_coords([self.get_square(sq)])[0]
            if piece:
                if piece.color != self.white:
                    self.known_presences[r][c] = 1
                    j = -1
                    if piece == reconchess.chess.PAWN:
                        self.set_then_normalize(5, r, c, 1)
                        j=5
                    elif piece == reconchess.chess.KNIGHT:
                        self.set_then_normalize(3, r, c, 1)
                        j=3
                    elif piece == reconchess.chess.BISHOP:
                        self.set_then_normalize(4, r, c, 1)
                        j=4
                    elif piece == reconchess.chess.ROOK:
                        self.set_then_normalize(2, r, c, 1)
                        j=2
                    elif piece == reconchess.chess.QUEEN:
                        self.set_then_normalize(1, r, c, 1)
                        j=1
                    elif piece == reconchess.chess.KING:
                        self.set_then_normalize(0, r, c, 1)
                        j=0
                    for i in range(6):
                        if i != j:
                            self.set_then_normalize(i, r, c, 0)
                else:
                    self.known_absences[r][c] = 1
            else:
                self.known_absences[r][c] = 1


    def capture(self, square: reconchess.Square):
        sq = self.get_square(square)
        r, c = convert_squares_to_coords([sq])[0]
        self.psuedo_presences[r][c] = 0
        self.psuedo_absences[r][c] = 1
        # zero out square for each piece in opp_beliefs
        # for each piece with a nonzero probabilitiy, normalize other squares probabilities for that piece
        for index in range(6):
            if self.opp_beliefs[index][r][c]:
                self.set_then_normalize(index, r, c, 0)

        self.num_opp_pieces -= 1
        
    def apply_impl(self, req_move: Optional[reconchess.chess.Move], taken_move: Optional[reconchess.chess.Move]):
        # assume the piece has not been moved yet (on our local board)!

        # if the piece was a Rook, Bishop, or Queen, set all probabilities in its path to zero,
        # (capture will already be taken care of)

        # if moving piece as a pawn and it tried to capture, there's no piece to capture on that square
        # if moving piece was a pawn and it tried to move forward, there is a piece on the square it tried to move forward to

        # get piece that is moving
        req_move = reconchess.chess.Move(self.get_square(req_move.from_square), self.get_square(req_move.to_square), req_move.promotion)
        if taken_move:
            taken_move = reconchess.chess.Move(self.get_square(taken_move.from_square), self.get_square(taken_move.to_square), taken_move.promotion)

        piece = self.board.piece_at(req_move.from_square)
        
        (from_r, from_c), (to_r, to_c) = convert_squares_to_coords([req_move.from_square, req_move.to_square])

        if piece.piece_type == reconchess.chess.PAWN:
            if from_c != to_c:
                self.psuedo_absences[to_r][to_c] = 1
            else:
                if taken_move is None:
                    self.psuedo_presences[from_r + (1 if self.white else -1)][from_c] = 1
                else:
                    taken_to_r, taken_to_c = convert_squares_to_coords(taken_move.to_square)[0]
                    self.psuedo_presences[taken_to_r + (1 if self.white else -1)][taken_to_c] = 1

    # assumes row and column are already normalized!
    def set_then_normalize(self, index: int, r: int, c: int, new_value: float) -> None:
        diff = new_value - self.opp_board[index][r][c]
        prob_sum = np.sum(self.opp_board[index])
        new_sum = prob_sum + diff
        coeff = max(prob_sum / new_sum, 0) if new_sum else 0
        self.opp_board[index] = np.multiply(self.opp_board[index], coeff)
        self.opp_board[index][r][c] = new_value