from core.config import GameConfig
from core.environment import CardGameEnv
from ui.terminal import run_terminal_game

config = GameConfig(num_players=2, num_decks=1, num_rounds=3, num_cards=2)
env = CardGameEnv(config, seed=42)
run_terminal_game(env)
