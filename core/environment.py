import random
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from core.cards import create_standard_deck
from core.state import GameState, RoundState, PlayerState 
from core.scoring import compute_player_score
from core.actions import apply_draw_action, apply_discard_action, apply_flip_action

class CardGameEnv:
    def __init__(self, config, seed: int = 42):
        self.config = config
        self.reset(seed)
        self.rng = random.Random(seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()

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
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        draw_action, discard_action, flip_action = actions

        # Perform draw -> discard -> flip using core action helpers
        drawn_cards = apply_draw_action(self, draw_action)
        apply_discard_action(self, drawn_cards, discard_action)
        apply_flip_action(self, flip_action)

        # Advance turn
        self.advance_turn()

        # If all players have all cards face up, end the round
        round_state.complete = all(p.is_face_up_all() for p in self.player_state)
        reward = -compute_player_score(player, style='known')

        # info = {"round_ended": round_ended}
        info = {}

        # return self._get_observation(), reward, round_state.complete, info
        return None, reward, round_state.complete, info

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reinitialize the environment. If seed is provided, reseed RNGs."""
        self.rng = random.Random(seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()
        # return self._get_observation()

    def initialize_game(self):
        self.game_state = GameState(
            current_player=0,
            round_number=0
        )
    
    def initialize_round(self):
        deck = create_standard_deck(self.config.num_decks)
        self.rng.shuffle(deck)

        self.round_state = RoundState(
            draw_deck=deck,
            discard_pile=[deck.pop()]
        )
        for player in self.player_state:
            player.reset([self.round_state.draw_deck.pop() for _ in range(self.config.cards_per_player)])
    
    def initialize_players(self):
        self.player_state = [
            PlayerState(i, self.config.cards_per_player) for i in range(self.config.num_players)
        ]

    def advance_turn(self):
        self.game_state.current_player = (self.game_state.current_player + 1) % self.config.num_players

    def advance_round(self):
        for player in self.player_state:
            score = compute_player_score(player, style='flipped')
            player.individual_scores.append(score)
        self.game_state.round_number += 1
        if self.game_state.round_number >= self.config.num_rounds:
            self.game_state.complete = True