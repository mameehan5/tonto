

from core.config import GameConfig
from core.environment import CardGameEnv
from ui.terminal import run_terminal_game

config = GameConfig(num_players=1, num_decks=1, cards_per_player=4, num_rounds=3)
env = CardGameEnv(config)
run_terminal_game(env)
