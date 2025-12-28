import random
from core.cards import create_standard_deck
from core.state import PlayerState, GameState
from core.scoring import compute_player_score

class CardGameEnv:
    def __init__(self, config):
        self.config = config
        self.initialize_game()
        self.initialize_players()

    def initialize_game(self):
        rng = random.Random(self.config.seed)
        deck = create_standard_deck(self.config.num_decks)
        rng.shuffle(deck)

        self.state = GameState(
            players=[],
            draw_deck=deck,
            discard_pile=[deck.pop()],
            current_player=0,
            round_number=0,
            rng=rng
        )
    
    def initialize_round(self):
        self.state.current_player = 0
        self.state.draw_deck = create_standard_deck(self.config.num_decks)
        self.state.rng.shuffle(self.state.draw_deck)
        self.state.discard_pile = [self.state.draw_deck.pop()]
        for player in self.state.players:
            player.cards = [self.state.draw_deck.pop() for _ in range(self.config.cards_per_player)]
            player.face_up = [False] * self.config.cards_per_player
            player.private_seen = [True] * (self.config.cards_per_player // 2) + [False] * (self.config.cards_per_player // 2)
    
    def initialize_players(self):
        players = []
        for _ in range(self.config.num_players):
            hand = [self.state.draw_deck.pop() for _ in range(self.config.cards_per_player)]
            players.append(PlayerState(hand))
        self.state.players = players

    def advance_turn(self, discarded_cards=[]):
        self.state.discard_pile.extend(discarded_cards)
        self.state.current_player = (self.state.current_player + 1) % self.config.num_players

    def advance_round(self):
        self.state.round_number += 1
        for player in self.state.players:
            score = compute_player_score(player.cards)
            player.individual_scores.append(score)
        self.initialize_round()