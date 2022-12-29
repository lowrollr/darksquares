import reconchess
from reconchess.bots.attacker_bot import AttackerBot
from darksquares import DarkSquaresBot


player1 = DarkSquaresBot()
player2 = AttackerBot()

reconchess.play_local_game(player1, player2)

