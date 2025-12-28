from dataclasses import dataclass

@dataclass
class GameConfig:
    num_players: int = 2
    num_decks: int = 1
    cards_per_player: int = 4
    num_rounds: int = 3
    seed: int = 42
