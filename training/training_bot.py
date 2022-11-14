



import collections
from datetime import datetime
import multiprocessing as mp
from random import choice, random
from typing import List, Optional, Tuple
import chess

import torch
from greatwhite import GreatWhiteBot
from net import BeliefNet
from training.training_evaluator import TrainingEvaluator
from reconchess.game import LocalGame
from reconchess.bots.attacker_bot import AttackerBot
from reconchess.bots.random_bot import RandomBot
from reconchess.bots.trout_bot import TroutBot
from reconchess.play import play_turn, notify_opponent_move_results
from utils.lc0 import LeelaWrapper
import numpy as np
from state import ID_MAPPING










# turn flow
# 1. play opponet's turn, notify opponet move results and update belief state -> add converted belief state to batch
# 2. (NONLOCAL) get belief net output -> return output to each evaluator
# 3. update/normalize belief state w/ belief net output + sense + update belief net state
# 4. (NONLOCAL) get belief net output -> add converted belief state to batch
# 6. update belief state and return best move

class TrainingBot(GreatWhiteBot):
    def __init__(self, leela) -> None:
        super().__init__(TrainingEvaluator(leela))

class TrainingGame(LocalGame):
    def __init__(self, game_id, leela, seconds_per_player: Optional[float] = None, seconds_increment: Optional[float] = None, reversible_moves_limit: Optional[int] = 100, full_turn_limit: Optional[int] = None):
        super().__init__(seconds_per_player, seconds_increment, reversible_moves_limit, full_turn_limit)
        self.game_id = game_id
        self.us = TrainingBot(leela)
        self.them = choice([AttackerBot, RandomBot, TroutBot])()
        
        self.we_play_white = False
        if random() > 0.5:
            self.we_play_white = True
        self.us.handle_game_start(color=chess.WHITE if self.we_play_white else chess.BLACK, board=self.board.copy(), opponent_name=self.them.__class__.__name__)
        self.them.handle_game_start(color=chess.BLACK if self.we_play_white else chess.WHITE, board=self.board.copy(), opponent_name=self.us.__class__.__name__)
        self.start()

    def store_game_result(self):
        pass

# step 1
def pre_sense(game: TrainingGame):
    if game.turn != game.we_play_white:
        # play opponet's turn (except when we play first of course)
        play_turn(game, game.them, end_turn_last=True)
    notify_opponent_move_results(game, game.us)
    return game.us.beliefs.to_nn_input()


def get_bn_output(model, optimizer, input, actual):
    input = np.stack(input)
    input = torch.from_numpy(input).to(model.device)
    print(actual)
    out = model(input)
    print('got output')
    loss = model.loss_fn(input, out, actual)
    print('computed loss')
    loss.backward()
    print('propogated loss')
    optimizer.step()
    print('ran optimizer')
    return list(out.detach().numpy())

# step 3
def sense(game: TrainingGame, model_output):
    sense_square = game.us.get_sense_square(model_output, game.us.beliefs)
    sense_result = game.sense(sense_square)
    game.us.handle_sense_result(sense_result)



# step 5
def post_sense(game: TrainingGame, model_output):
    move = game.us.get_best_move(model_output, game.us.beliefs)
    requested_move, taken_move, opt_enemy_capture_square = game.move(move)
    

    game.us.handle_move_result(requested_move, taken_move,
                              opt_enemy_capture_square is not None, opt_enemy_capture_square)
    game.end_turn()

# instantiate static pool of workers that take jobs from the job queue
# designate a 'model' worker that handles running input through the model

class ModelContext:
    def __init__(self) -> None:
        self.model = BeliefNet(22,8)
        self.optimizer = torch.optim.Adam(self.model.parameters())




        
    

    # State Machine here is a little complex...

    # 1. We collect model input until our batch size is reached
    # 2. If we do not have enough model input, create more games to generate more input
    # 3. Once adequate model input is collected, get results and map back to games requesting for
    # 4. continue with next fn in flow

def start_training(batch_size=1024, batches=1000):
    num_procs = 2
    with mp.Pool(processes=num_procs) as pool:
        manager = mp.Manager()
        input = manager.Queue()
        completed_batches = manager.Value('i', 0)
        output_channels = [manager.Queue() for _ in range(num_procs)]
        args = []
        for i in range(num_procs):
            ctx = None
            out_channels = None
            if i == 0:
                ctx = ModelContext()
                out_channels = output_channels
            args.append((i, input, output_channels[i], out_channels, completed_batches, batch_size, batches, ctx))
        pool.starmap(train, args)

# main function for training worker
# maintains local game/job queues
def train(id, input, output, output_channels, completed_batches, batch_size, batches, ctx=None):
    # one engine per process
    engine = LeelaWrapper()
    games_awaiting_output = dict()
    last_game_id = 0
    jobs = collections.deque()
    while batches > completed_batches.value:
        
        if ctx:
            if input.qsize() > batch_size:
                print('running batch! input queue size =', input.qsize())
                run_batch(input, ctx.model, ctx.optimizer, batch_size, output_channels)
                completed_batches.value += 1
        if not output.empty():
            while not output.empty():
                # spawn new jobs with output
                model_out, game_id = output.get()
                game, sense_next = games_awaiting_output[game_id]
                if sense_next:
                    jobs.append((sense, (game, model_out)))
                else:
                    jobs.append((sense, (game, model_out)))
                games_awaiting_output.pop(game_id)

        elif jobs:
            
            fn, args = jobs.popleft()
            result = fn(*args)
            game = args[0]
            if fn == pre_sense:
                games_awaiting_output[game.game_id] = (game, True)
                input.put((result, convert_board_to_target(game), game.game_id, id))
            elif fn == sense:
                games_awaiting_output[game.game_id] = (game, False)
                input.put((result, convert_board_to_target(game), game.game_id, id))
            elif fn == post_sense:
                jobs.append((pre_sense, (game,)))
        else:
            jobs.append((pre_sense, (TrainingGame(last_game_id, engine),)))
            last_game_id += 1
    

def run_batch(input, model, optimizer, batch_size, output_channels):
    actual = []
    batch_input = []
    ids = []
    while len(actual) < batch_size:
        input_board, actual_board, process_id, game_id = input.get()
        actual.append(actual_board)
        batch_input.append(input_board)
        ids.append((process_id, game_id))
        print(len(actual))
    print('accumulated batch!')
    result = get_bn_output(model, optimizer, batch_input, actual)
    print('got result')
    for i,r in enumerate(result):
        p_id, game_id = ids[i]
        output_channels[p_id].put((r, game_id))
    print('done with batch')

def convert_board_to_target(game: TrainingGame) -> np.ndarray:
    # PIECE LOCATIONS
    piece_locs = np.zeros(shape=(6,8,8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            sq = (r * 8) + c
            piece = game.board.piece_at(sq)
            if piece and piece.color != game.we_play_white: 
                index = ID_MAPPING[piece.piece_type]
                piece_locs[index][r][c] = 1
    # EN PASSANT

    # CASTLING RIGHTS
    
    return piece_locs


