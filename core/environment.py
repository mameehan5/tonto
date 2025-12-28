import random
from typing import Tuple, List, Dict, Any, Optional

from core.cards import create_standard_deck
from core.state import PlayerState, GameState
from core.scoring import compute_player_score
from core.actions import apply_draw_action, apply_discard_action, apply_flip_action

class CardGameEnv:
    def __init__(self, config):
        self.config = config
        self.initialize_game()
        self.initialize_players()

    def reset(self, seed: int = None) -> Dict[str, Any]:
        """Reset the environment. Returns an initial observation dict.

        If `seed` is provided, it overrides `self.config.seed` for this reset.
        """
        self.initialize_game(seed=seed)
        self.initialize_players()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        """Return a serializable observation of the current state."""
        state = self.state
        obs_players = []
        for p in state.players:
            obs_players.append({
                "cards": list(p.cards),
                "face_up": list(p.face_up),
                "private_seen": list(p.private_seen),
                "individual_scores": list(p.individual_scores),
            })

        return {
            "current_player": state.current_player,
            "round_number": state.round_number,
            "draw_deck_size": len(state.draw_deck),
            "discard_pile": state.discard_pile,
            "players": obs_players,
        }

    def step(self, actions: Tuple) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply a composite action (draw_action, discard_action, flip_action).

        Returns: (observation, reward, done, info)
        """
        player = self.state.players[self.state.current_player]
        draw_action, discard_action, flip_action = actions

        # Perform draw -> discard -> flip using core action helpers
        drawn_cards = apply_draw_action(self, draw_action)
        apply_discard_action(self, drawn_cards, discard_action)
        apply_flip_action(self, flip_action)

        # Advance turn with discarded cards
        self.advance_turn()

        # If all players have all cards face up, end the round
        round_ended = all(p.all_face_up() for p in self.state.players)
        done = False
        reward = compute_player_score(player, style='known')

        if round_ended:
            # # Compute and record scores, then initialize next round
            # self.advance_round()
            # # Mark done when configured number of rounds completed
            # if self.state.round_number >= self.config.num_rounds:
            done = True

        info = {"round_ended": round_ended}

        return self._get_observation(), reward, done, info

    def initialize_game(self, seed: int = None):
        seed_to_use = self.config.seed if seed is None else seed
        rng = random.Random(seed_to_use)
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
        self.state.current_player = self.state.round_number % self.config.num_players
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

    def advance_turn(self):
        self.state.current_player = (self.state.current_player + 1) % self.config.num_players

    def advance_round(self):
        self.state.round_number += 1
        for player in self.state.players:
            score = compute_player_score(player, style='flipped')
            player.individual_scores.append(score)
        self.initialize_round()