


# script to generate synthetic data using lichess game positions

# 1. grab a random position from a lichess game
# 2. convert to BeliefState format
# 3. add random amount of noise∂
# 4. add to training data
import chess
import numpy as np
import scipy.stats
import chess.pgn
from state import ID_MAPPING, BeliefState
from utils.board import convert_squares_to_coords, mirror_move, get_squares_between_incl
import random
import time
import hickle


pgn = open('/Users/marshingjay/Downloads/lichess_db_standard_rated_2013-01.pgn')
game = chess.pgn.read_game(pgn)
start_time = time.time()

inputs = []
actuals = []
total_games = 121332
games = 0


def get_bitboard(board):
    opp_board = np.zeros(shape=(6, 8, 8))
    # write to belief state
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            piece_id = ID_MAPPING[piece.piece_type]
            if piece.color == chess.BLACK:
                r, c = convert_squares_to_coords([i])[0]
                opp_board[piece_id][r][c] = 1

    opp_en_passant = np.zeros(shape=(1,8))
    if board.ep_square is not None:
        r, c = convert_squares_to_coords([board.ep_square])[0]
        opp_en_passant[0][c] = 1

    opp_castle_q = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    opp_castle_k = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    return opp_board, opp_en_passant, (opp_castle_q, opp_castle_k)


def sense_board(board, sense_square):
    # sense 3x3 square centered around sense_square
    # sense_square is a chess square
    
    squares = [sense_square + 7, sense_square + 8, sense_square + 9, 
               sense_square - 1, sense_square, sense_square + 1, 
               sense_square - 9, sense_square - 8, sense_square - 7]

    return [(sq, board.piece_at(sq)) for sq in squares]


def apply_noise(state: BeliefState):
    lower = 0
    upper = 1
    mu = 0.1
    sigma = 0.1
    var = scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1)[0]
    # we want to apply noise to opponet beliefs
    state.opp_board += np.random.normal(0, var, size=(6,8,8))
    state.opp_board = np.clip(state.opp_board, 0, 1)
    state.opp_en_passant += np.random.normal(0, var, size=(8))
    state.opp_en_passant = np.clip(state.opp_en_passant, 0, 1)
    state.opp_castle_q = np.clip(np.random.normal(0, var), 0, 1)
    state.opp_castle_k = np.clip(np.random.normal(0, var), 0, 1)

count = 0

while game:
    board = chess.Board()
    state = BeliefState(playing_white=True)
    mirror_state = BeliefState(playing_white=True)
    mirror_board = chess.Board()
    mirror_board.turn = chess.BLACK
    for move in game.mainline_moves():
        if board.turn == chess.BLACK:
            state.update(*get_bitboard(board))
        if mirror_board.turn == chess.BLACK:
            mirror_state.update(*get_bitboard(mirror_board))

        mirror = mirror_move(move)
        capture = board.is_capture(move)
        mirror_capture = mirror_board.is_capture(mirror)

        en_passant = board.is_en_passant(move)
        mirror_en_passant = mirror_board.is_en_passant(mirror)
    

        board.push(move)
        mirror_board.push(mirror_move(move))
        

        if board.turn == chess.WHITE:
            # last move was made by opponet
            capture_square = None
            if en_passant:
                # row behind, so - 8
                capture_square = move.to_square + 8
            elif capture:
                capture_square = move.to_square
            state.opp_move(capture_square=capture_square)
            if bool(random.getrandbits(1)):
                # make sense action
                sense_sq = random.choice(range(36))
                sense_sq = (sense_sq // 6) * 8 + (sense_sq % 6) + 9
                sense_results = sense_board(board, sense_sq)
                state.set_ground_truth(sense_results)
            apply_noise(state)
            inputs.append(state.to_nn_input())
            actuals.append(get_bitboard(board))
            count += 1

        else:
            state.clear_information_gain()
            capture_square = None
            if en_passant:
                capture_square = move.to_square - 8
            elif capture:
                capture_square = move.to_square
            
            if capture_square:
                state.capture(capture_square)

            state.apply_move(move)
            state.board.push(chess.Move.from_uci('0000'))


        if mirror_board.turn == chess.WHITE:
            capture_square = None
            if mirror_en_passant:
                capture_square = mirror.to_square + 8
            elif mirror_capture:
                capture_square = mirror.to_square

            mirror_state.opp_move(capture_square=capture_square)
            if bool(random.getrandbits(1)):
                # make sense action
                sense_sq = random.choice(range(36))
                sense_sq = (sense_sq // 6) * 8 + (sense_sq % 6) + 9
                
                sense_results = sense_board(mirror_board, sense_sq)

                mirror_state.set_ground_truth(sense_results)
            apply_noise(mirror_state)
            inputs.append(mirror_state.to_nn_input())
            actuals.append(get_bitboard(mirror_board))
            count += 1
        else:
            mirror_state.clear_information_gain()
            capture_square = None
            if mirror_en_passant:
                capture_square = mirror.to_square - 8
            elif mirror_capture:
                capture_square = mirror.to_square
            
            if capture_square:
                mirror_state.capture(capture_square)

            mirror_state.apply_move(mirror)
            mirror_state.board.push(chess.Move.from_uci('0000'))
            
    print(f'Completed {count}, rate: {count / (time.time() - start_time)}/s')
    games += 1
    print(f'Completed {games} / {total_games}')
    game = chess.pgn.read_game(pgn)

hickle.dump(inputs, 'inputs.hkl')
hickle.dump(actuals, 'actuals.hkl')


            



        


    

