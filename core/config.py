from dataclasses import dataclass
from math import comb

@dataclass(frozen=True)
class GameConfig:
    num_players: int
    num_decks: int
    num_rounds: int
    num_cards: int = 4

    def __post_init__(self):
        assert type(self.num_cards) is int
        assert self.num_cards % 2 == 0
        assert 2 <= self.num_cards <= 20

        # Calculate number of possible actions: 
        #   draw from deck + pick up from discard options
        object.__setattr__(
            self, 
            'num_possible_actions', 
            1 + 2 * self.num_cards + sum(
                comb(self.num_cards, n) for n in range(1, self.num_cards + 1)
            )
        )

        # Other helpful attributes
        object.__setattr__(self, 'num_total_cards', self.num_decks * 52)
        object.__setattr__(self, 'max_card_index', self.num_total_cards - 1)
        object.__setattr__(self, 'num_ranks', 13)
        object.__setattr__(self, 'num_suits', 4)

        object.__setattr__(self, 'min_possible_score', -5 * self.num_cards)
        object.__setattr__(self, 'max_possible_score', 10 * self.num_cards)

        object.__setattr__(self, 'max_num_cards', 20)
