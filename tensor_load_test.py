from tracemalloc import start
import torch
import time
import chess
import numpy as np

PIECE_ID_MAP = {
    (chess.PAWN, 0) : 0,
    (chess.PAWN, 1):  1,
    (chess.KNIGHT, 0): 2,
    (chess.KNIGHT, 1): 3,
    (chess.BISHOP, 0): 4,
    (chess.BISHOP, 1): 5,
    (chess.ROOK, 0): 6,
    (chess.ROOK, 1): 7,
    (chess.QUEEN, 0): 8,
    (chess.QUEEN, 1): 9,
    (chess.KING, 0): 10,
    (chess.KING, 1): 11,
}

def convert_to_nn_state(board: chess.Board, reps=0):
    # 12 piece planes (6 piece types per player)
    data_tensor = np.zeros(shape=(20,8,8), dtype=np.float32)
    # num of halfmoves (for fifty move rule)
    data_tensor[12] = np.unpackbits(np.array([board.halfmove_clock], dtype=np.uint8), count=8)
    data_tensor[13] = np.unpackbits(np.array([reps], dtype=np.uint8), count=8)
    if board.has_kingside_castling_rights(1):
        data_tensor[14] = 1
    if board.has_queenside_castling_rights(1):
        data_tensor[15] = 1
    if board.has_kingside_castling_rights(0):
        data_tensor[16] = 1
    if board.has_queenside_castling_rights(0):
        data_tensor[17] = 1
    if board.has_legal_en_passant():
        data_tensor[18] = 1
    if board.turn:
        data_tensor[19] = 1
    for sq, piece in board.piece_map().items():
        v = PIECE_ID_MAP[(piece.piece_type, piece.color)]
        r, c = sq // 8, sq % 8
        data_tensor[v][r][c] = 1
    # so we can hash 
    data_tensor.flags.writeable = False
    return data_tensor

iterations = 10000
board = chess.Board()
print("Trial 1: Create Tensor on CPU, load onto GPU each iter")
start_time = time.time()
for i in range(iterations):
    tensor = torch.from_numpy(convert_to_nn_state(board)).to('cuda:0')

print("Time elapsed: ", time.time() - start_time)

print("Trial 2: Create Tensor on GPU, update with new values each iter")
start_time = time.time()


