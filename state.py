
from typing import List, Optional, Tuple
from matplotlib.pyplot import pie
import numpy as np
import reconchess
import torch



ID_MAPPING = {
    reconchess.chess.KING:   0,
    reconchess.chess.QUEEN:  1,
    reconchess.chess.ROOK:   2,
    reconchess.chess.KNIGHT: 3,
    reconchess.chess.BISHOP: 4,
    reconchess.chess.PAWN:   5
}

        

class BeliefState:
    def __init__(self, playing_white=True) -> None:
        self.board = reconchess.chess.Board('8/8/8/8/8/8/PPPPPPPP/RNBQKBNR')
        if not playing_white:
            self.board = reconchess.chess.Board('rnbqkbnr/pppppppp/8/8/8/8/8/8')
        self.absences = np.zeros(shape=(2,8,8))
        self.presences = np.zeros(shape=(2,8,8))
        self.num_moves = 0
        self.num_opp_pieces = 16
        self.opp_beliefs: np.ndarray = np.zeros(shape=(8,8,8))
        self.white = playing_white

        self.init_opp_beliefs()
    
    def init_opp_beliefs(self):
        for piece, i in ID_MAPPING.items():
            if piece == reconchess.chess.PAWN:
                self.opp_beliefs[i][6 if self.white else 1] = 1
            elif piece == reconchess.chess.KNIGHT:
                self.opp_beliefs[i][7 if self.white else 0][1] = 1
                self.opp_beliefs[i][7 if self.white else 0][6] = 1
            elif piece == reconchess.chess.BISHOP:
                self.opp_beliefs[i][7 if self.white else 0][2] = 1
                self.opp_beliefs[i][7 if self.white else 0][5] = 1
            elif piece == reconchess.chess.ROOK:
                self.opp_beliefs[i][7 if self.white else 0][0] = 1
                self.opp_beliefs[i][7 if self.white else 0][7] = 1
            elif piece == reconchess.chess.QUEEN:
                self.opp_beliefs[i][7 if self.white else 0][3] = 1
            elif piece == reconchess.chess.KING:
                self.opp_beliefs[i][7 if self.white else 0][4] = 1
        self.opp_beliefs[6] = 0
        self.opp_beliefs[7] = 1
        self.absences[0][:2] = 1
        self.presences[0][6:] = 1

    def to_nn_input(self, moved_last=False) -> np.ndarray:
        data_tensor = np.zeros(shape=(22,8,8), dtype=np.float32)

        # -- INFORMATION GAIN -- #
        data_tensor[0:2] = self.absences
        data_tensor[2:4] = self.presences
        data_tensor[4] = np.unpackbits(np.array([self.num_moves], dtype=np.uint8), count=8)
        data_tensor[5] = np.unpackbits(np.array([self.num_opp_pieces], dtype=np.uint8), count=8)

        # -- CASTLING -- # 
        if self.board.has_kingside_castling_rights(self.white):
            data_tensor[6,:,0:4] = 1
        if self.board.has_queenside_castling_rights(self.white):
            data_tensor[6,:,4:] = 1

        # -- EN PASSANT -- # jesus this code is bad
        if moved_last:
            # if our last move was a pawn and it moved two spaces, flag en passant on that rank
            if len(self.board.move_stack):
                move = self.board.move_stack[-1]
                fr_sq, to_sq = move.from_square, move.to_square
                if self.board.piece_at(fr_sq) == reconchess.chess.PAWN and abs(fr_sq - to_sq) == 16:
                    file = fr_sq % 8
                    data_tensor[7,:,file] = 1
        else:
            if len(self.board.move_stack) > 1:
                move = self.board.move_stack[-2]
                fr_sq, to_sq = move.from_square, move.to_square
                if self.board.piece_at(fr_sq) == reconchess.chess.PAWN and abs(fr_sq - to_sq) == 16:
                    file = fr_sq % 8
                    data_tensor[7,:,file] = 1

        # -- OUR PIECES -- #
        for sq, piece in self.board.piece_map().items():
            v = ID_MAPPING[piece.piece_type] + 8
            r, c = sq // 8, sq % 8
            data_tensor[v][r][c] = 1

        # -- ENEMY PIECES -- #
        data_tensor[14:] = self.opp_beliefs
        
        return data_tensor
        
        
    def reset_absences_presences(self) -> None:
        self.absences = np.zeros(shape=(2,8,8))
        self.presences = np.zeros(shape=(2,8,8))

    def set_absences(self, index):
        for r in range(8):
            for c in range(8):
                sq = (r * 8) + c
                if self.board.piece_at(sq):
                    self.absences[index][r][c] = 1

    def opp_move(self, capture_square: Optional[reconchess.Square]) -> None:
        if capture_square:
            rank, file = capture_square // 8, capture_square % 8
            self.board.remove_piece_at(capture_square)
            self.presences[0][rank][file] = 1
        self.set_absences(0)

    def update(self, probs, passant, castle):
        print(probs)
        self.opp_beliefs[:6] = probs
        for i in range(8):
            self.opp_beliefs[6][i] = passant[i]
        self.opp_beliefs[7,:4] = castle[0]
        self.opp_beliefs[7,4:] = castle[1]
        
    # apply the move to our board, and make sure any gleaned information about our opponet's pieces makes it into our belief state
    def apply_move(self, move: reconchess.chess.Move) -> None:
        self.reset_absences_presences()
        from_sq, to_sq = self.convert_move_from_to(move)
        in_between_squares = []
        print(self.board, move)
        piece = self.board.piece_at(move.from_square)
        # if moving a grounded multi-square piece (Bishop/Rook/Queen), we know all squares in between our starting and ending square were unoccupied by an opposing piece
        # so we can set them to zero and adjust all other probabilities on the board to maintain implied piece count
        if piece.piece_type in {reconchess.chess.ROOK, reconchess.chess.QUEEN, reconchess.chess.BISHOP}:
            in_between_squares = self.get_squares_between_incl(from_sq, to_sq)
        elif piece.piece_type == reconchess.chess.KING:
            # check if this was a castle move, if so the squares between the rook and the king were unoccupied
            if move.xboard() == 'O-O-O':
                in_between_squares = self.get_squares_between_incl(from_sq, from_sq - 3)
            elif move.xboard() == 'O-O':
                in_between_squares = self.get_squares_between_incl(from_sq, from_sq + 2)
            else:
                in_between_squares = [from_sq, to_sq]

        elif piece.piece_type == reconchess.chess.PAWN:
            # check if moved 2 spaces, space between was unoccuppied if so
            if abs(from_sq - to_sq) == 16:
                in_between_squares = self.get_squares_between_incl(from_sq, to_sq)
            else:
                in_between_squares = [from_sq, to_sq]
        else:
            # knights go where they want to
            in_between_squares = [from_sq, to_sq]

        self.board.push(move)
        self.set_absences(1)
        for s in in_between_squares:
            rank, file = s // 8, s % 8
            self.absences[1][rank][file] = 1

    


    def set_ground_truth(self, truth: List[Tuple[reconchess.Square, Optional[reconchess.chess.Piece]]]):
        for sq, piece in truth:
            r, c = sq // 8, sq % 8
        
            if piece:
                if piece.color != self.white:
                    self.presences[0][r][c] = 1
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
                    self.absences[0][r][c] = 1
            else:
                self.absences[0][r][c] = 1


    def capture(self, square: reconchess.Square):
        r,c = square // 8, square % 8
        self.presences[1][r][c] = 0
        self.absences[1][r][c] = 1
        # zero out square for each piece in opp_beliefs
        # for each piece with a nonzero probabilitiy, normalize other squares probabilities for that piece
        for index in range(6):
            if self.opp_beliefs[index][r][c]:
                self.set_then_normalize(index, r, c, 0)

        self.num_opp_pieces -= 1
        
    def apply_impl(self, req_move: Optional[reconchess.chess.Move], taken_move: Optional[reconchess.chess.Move]):
        # assume the piece has not been moved yet!

        # if the piece was a Rook, Bishop, or Queen, set all probabilities in its path to zero,
        # (capture will already be taken care of)

        # if moving piece as a pawn and it tried to capture, there's no piece to capture on that square
        # if moving piece was a pawn and it tried to move forward, there is a piece on the square it tried to move forward to

        piece = self.board.piece_at(req_move.from_square)
        from_r, from_c = req_move.from_square // 8, req_move.from_square % 8
        to_r, to_c = req_move.to_square // 8, req_move.to_square % 8

        if piece.piece_type == reconchess.chess.PAWN:
            if from_c != to_c:
                self.absences[1][to_r][to_c] = 1
            else:
                if taken_move is None:
                    self.presences[1][from_r + (1 if self.white else -1)][from_c] = 1
                else:
                    taken_to_r, taken_to_c = taken_move.to_square // 8, taken_move.to_square % 8
                    self.presences[1][taken_to_r + (1 if self.white else -1)][taken_to_c] = 1

    def set_then_normalize(self, index: int, r: int, c: int, new_value: float) -> None:
        diff = new_value - self.opp_beliefs[index][r][c]
        prob_sum = np.sum(self.opp_beliefs[index])
        new_sum = prob_sum + diff
        coeff = max(prob_sum / new_sum, 0) if new_sum else 0
        self.opp_beliefs[index] = np.multiply(self.opp_beliefs[index], coeff)
        self.opp_beliefs[index][r][c] = new_value
            
    def convert_move_from_to(self, move: reconchess.chess.Move):
        if self.white:
            return move.from_square, move.to_square
        else:
            return (56 + (move.from_square % 8)) - move.from_square, (56 + (move.to_square % 8)) - move.to_square

    @staticmethod
    def get_squares_between_incl(start: reconchess.Square, end: reconchess.Square) -> List[reconchess.Square]:
        start_rank = start // 8
        end_rank = end // 8
        start_file = start % 8
        end_file = end % 8
        # TODO: handle going backwards
        if start_rank == end_rank:
            # across rank
            return [(8 * start_rank) + i for i in range(start_file, end_file)]
        elif start_file == end_file:
            # up/down file
            return [(8 * i) + start_file for i in range(start_rank, end_rank)]
        else:
            # diagonal
            return [(8 * (start_rank + i)) + (start_file + i) for i in range(start_rank, end_rank)]