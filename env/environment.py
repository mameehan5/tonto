import random
from core.cards import create_standard_deck
from env.state import PlayerState, GameState
from core.phases import TurnPhase
from core.engine import apply_action

class CardGameEnv:
    def __init__(self, config):
        self.config = config
        self.pending_draw = None
        self.pending_source = None
        self.pending_count = 0
        self.just_flipped = []

        # Initialize first round automatically
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
            # phase=TurnPhase.DRAW,
            round_number=0,
            rng=rng
        )
    
    def initialize_players(self):
        players = []
        for _ in range(self.config.num_players):
            hand = [self.state.draw_deck.pop() for _ in range(self.config.cards_per_player)]
            players.append(PlayerState(hand))
        self.state.players = players

    def advance_turn(self):
        if all(len(p.face_up) == len(p.cards) for p in self.players):
            print("ROUND OVER")
            return
        self.state.current_player = (self.state.current_player - 1) % len(self.players)
        self.state.phase = TurnPhase.DRAW
