from dataclasses import dataclass
from math import comb

@dataclass(frozen=True)
class GameConfig:
    num_players: int
    num_decks: int
    cards_per_player: int
    num_rounds: int

    def __post_init__(self):
        # Calculate number of possible actions: 
        #   draw from deck + pick up from discard options
        object.__setattr__(
            self, 
            'num_possible_actions', 
            1 + 2 * self.cards_per_player + sum(
                comb(self.cards_per_player, n) for n in range(1, self.cards_per_player + 1)
            )
        )

        # Other helpful attributes
        object.__setattr__(self, 'total_cards', self.num_decks * 52)
        object.__setattr__(self, 'max_card_index', self.total_cards - 1)
        object.__setattr__(self, 'num_ranks', 13)
        object.__setattr__(self, 'num_suits', 4)

        object.__setattr__(self, 'min_possible_score', -5 * self.cards_per_player)
        object.__setattr__(self, 'max_possible_score', 10 * self.cards_per_player)
