import random
from core.cards import create_standard_deck
from env.state import PlayerState, GameState
from core.phases import TurnPhase
from core.engine import apply_action

class CardGameEnv:
    def __init__(self, config):
        self.config = config
        self.players = [PlayerState([]) for _ in range(config.num_players)]
        self.state: GameState = None
        self.pending_draw = None
        self.pending_source = None
        self.pending_count = 0
        self.just_flipped = []

        # Initialize first round automatically
        self.init_round(0)

    def init_round(self, round_number):
        rng = random.Random(self.config.seed + round_number)
        deck = create_standard_deck(self.config.num_decks)
        rng.shuffle(deck)

        for player in self.players:
            player.cards = [deck.pop() for _ in range(self.config.cards_per_player)]
            player.face_up.clear()
            player.private_seen.clear()

        self.state = GameState(
            players=self.players,
            draw_deck=deck,
            discard_pile=[deck.pop()],
            current_player=0,
            phase=TurnPhase.DRAW,
            round_number=round_number,
            rng=rng
        )

    def advance_turn(self):
        if all(len(p.face_up) == len(p.cards) for p in self.players):
            print("ROUND OVER")
            return
        self.state.current_player = (self.state.current_player - 1) % len(self.players)
        self.state.phase = TurnPhase.DRAW
