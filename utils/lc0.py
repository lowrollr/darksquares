import os
from typing import Tuple
import chess
import chess.engine

leela_dot_exe = os.getenv('LC0_EXE')
leela_backend = os.getenv('LC0_BACKEND')





class LeelaWrapper():
    def __init__(self) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(leela_dot_exe)
        self.engine.configure({'backend': leela_backend})

    def get_move_probabilities(self, board: chess.Board, time_limit=1, multipv=200):
        analysis = self.engine.analysis(board, chess.engine.Limit(time_limit), multipv=multipv, options={'VerboseMoveStats': True}, info=chess.engine.INFO_BASIC)
        move_probabilities = dict()
        for m in analysis:
            # this is a pretty dumb way to parse the analysis dump but is probably good enough for now
            if m.get('string'):
                move, prob = self.parse_leela_custom_output(m['string'])
                if move != 'node':
                    move_probabilities[move] = prob

        return move_probabilities

    def get_engine_centipawn_eval(self, board, time_limit=1) -> int:
        score: chess.engine.PovScore = self.engine.analyse(board, limit=chess.engine.Limit(time_limit))['score']
        return score.white().score(mate_score=100000) if board.turn else score.black().score(mate_score=100000)

    @staticmethod
    def parse_leela_custom_output(text: str) -> Tuple[str, float]:
        move = text.split(' ')[0]
        prob = float(text.split(')')[2][-6:-1]) / 100
        return (move, prob)

    def quit(self):
        self.engine.quit()