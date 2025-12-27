from core.config import GameConfig
from env.environment import CardGameEnv
from ui.terminal import run_terminal_game

config = GameConfig(num_players=2, num_decks=1, cards_per_player=4, num_rounds=3, seed=42)
env = CardGameEnv(config)

# for round_number in range(config.num_rounds):
#     env.init_round(round_number)
run_terminal_game(env)

print("\nGame over! All rounds completed.")
