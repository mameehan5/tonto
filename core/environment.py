import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Any, Optional

from core.cards import create_standard_deck, card_to_int
from core.state import GameState, RoundState, PlayerState 
from core.scoring import compute_player_score
from core.actions import apply_draw_action, apply_discard_action, apply_flip_action

class CardGameEnv:
    def __init__(self, config, seed: int = 42):
        self.config = config
        self.reset(seed)

        max_card_index = self.config.num_decks * 52 - 1
        self.observation_space = spaces.Dict({
            "player_cards": spaces.Box(
                low=-1, 
                high=max_card_index,
                shape=(self.config.num_players, self.config.cards_per_player),
                dtype=np.int32
            ),
            "unknown_card_size": spaces.Box(
                low=0, 
                high=max_card_index,
                shape=(), 
                dtype=np.int32
            ),
            "discarded_cards": spaces.Box(
                low=-1, 
                high=max_card_index, 
                shape=(self.config.cards_per_player,), 
                dtype=np.int32
            ),
            "current_player": spaces.Discrete(
                self.config.num_players
            ),
        })

    def _get_observation(self) -> Dict[str, Any]:
        """Return a gym-compatible observation that matches self.observation_space."""
        # allocate arrays with expected shapes/dtypes from the configured space
        num_players = self.config.num_players
        num_cards = self.config.cards_per_player

        # face_up = np.zeros((num_players, num_cards), dtype=np.int8)

        # Set player cards
        player_cards = np.full((num_players, num_cards), -1, dtype=np.int32)
        for ip, player in enumerate(self.player_state):
            for ic, (c,up) in enumerate(zip(player.cards, player.face_up)):
                if up:
                    player_cards[ip, ic] = card_to_int(c)
        
        # Set discarded cards
        discarded_cards = np.full((num_cards), -1, dtype=np.int32)
        for i in range(num_cards):
            if i < len(self.round_state.discard_pile):
                discarded_cards[-i] = card_to_int(self.round_state.discard_pile[-i])
        # discarded_cards[:len(self.round_state.discard_pile[-num_cards:])] = [card_to_int(c) for c in self.round_state.discard_pile[-num_cards:]]

        # Calculate unknown card size
        unknown_card_size = len(self.round_state.draw_deck)
        for player in self.player_state:
            unknown_card_size += player.num_face_down

        return {
            "player_cards": player_cards,
            "unknown_card_size": np.int32(unknown_card_size),
            "discarded_cards": discarded_cards,
            "current_player": int(self.game_state.current_player),
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

        return self._get_observation(), reward, round_state.complete, info

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reinitialize the environment. If seed is provided, reseed RNGs."""
        self.rng = random.Random(seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()
        return self._get_observation()

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