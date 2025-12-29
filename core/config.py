from math import comb

class GameConfig:
    def __init__(
        self, 
        num_players: int=2, 
        num_decks: int=1, 
        cards_per_player: int=4, 
        num_rounds: int=3
    ):
        self.num_players = num_players
        self.num_decks = num_decks
        self.cards_per_player = cards_per_player
        self.num_rounds = num_rounds

        # Calculate number of possible actions
        #   draw from deck + pick up from discard options
        self.num_possible_actions = 2 * self.cards_per_player
        for n in range(1, self.cards_per_player + 1):
            self.num_possible_actions += comb(self.cards_per_player, n)
        
        # Calculate total number of cards
        self.total_cards = self.num_decks * 52
        self.max_card_index = self.total_cards - 1

        self.num_ranks = 13
        self.num_suits = 4