import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Any, Optional
from itertools import combinations

from math import comb

from core.cards import Card, create_standard_deck
# from core.cards import encode_card, decode_card
from core.state import GameState, RoundState, PlayerState 
from core.scoring import compute_player_score
from core.actions import apply_draw_action, apply_discard_action, apply_flip_action
from core.actions import ActionType, Action

class CardGameEnv(gym.Env):
    def __init__(self, config, seed: int = 42):
        self.config = config
        self.reset(seed=seed)

        # Check if encode/decode are consistent inverses
        for action_int in range(self.config.num_possible_actions):
            actions = self.decode_action(action_int)
            reencoded = self.encode_actions(actions)
            assert action_int == reencoded, f"Action encoding/decoding mismatch: {action_int} -> {actions} -> {reencoded}"

        self.observation_space = spaces.Dict({
            "player_card_ranks": spaces.MultiBinary(
                (
                    self.config.num_players, 
                    self.config.cards_per_player, 
                    self.config.num_ranks + 1 # +1 for face-down indicator
                )
            ),
            # "unknown_card_size": spaces.Box(
            #     low=0, 
            #     high=self.config.total_cards,
            #     shape=(1,), 
            #     dtype=np.int32
            # ),
            # "discarded_cards": spaces.MultiBinary(
            #     (
            #         self.config.cards_per_player, 
            #         self.config.total_cards
            #     )
            # ),
            # "current_player": spaces.Discrete(
            #     self.config.num_players
            # ),
            # "legal_actions_mask": spaces.MultiBinary(
            #     (self.config.num_possible_actions,)
            # ),
            # Need to add top of card pile
            # Need to add mask based on legal actions if draw card is known
        })
        self.action_space = spaces.Discrete(self.config.num_possible_actions)

    def _get_obs(self) -> Dict[str, Any]:
        """Return a gym-compatible observation that matches self.observation_space."""
        # allocate arrays with expected shapes/dtypes from the configured space
        num_players = self.config.num_players
        num_cards = self.config.cards_per_player
        num_ranks = self.config.num_ranks

        # face_up = np.zeros((num_players, num_cards), dtype=np.int8)

        # Player card ranks
        player_card_ranks = np.zeros((num_players, num_cards, num_ranks + 1), dtype=np.int8)
        for ip, player in enumerate(self.player_state):
            for ic, (c,up) in enumerate(zip(player.cards, player.face_up)):
                if up is True:
                    player_card_ranks[ip, ic, c.rank] = 1
                else:
                    player_card_ranks[ip, ic, self.config.num_ranks] = 1  # face-down indicator
                
        # Set discarded cards
        discarded_cards = np.full((num_cards), -1, dtype=np.int32)
        for i in range(num_cards):
            if i < len(self.round_state.discard_pile):
                # discarded_cards[-i] = encode_card(self.round_state.discard_pile[-i])
                discarded_cards[-i] = self.round_state.discard_pile[-i].id
        # discarded_cards[:len(self.round_state.discard_pile[-num_cards:])] = [card_to_int(c) for c in self.round_state.discard_pile[-num_cards:]]

        # Calculate unknown card size
        unknown_card_size = len(self.round_state.draw_deck)
        for player in self.player_state:
            unknown_card_size += player.num_face_down

        return {
            "player_card_ranks": player_card_ranks,
            # "unknown_card_size": np.array([unknown_card_size], dtype=np.int32),
            # "discarded_cards": discarded_cards,
            # "current_player": int(self.game_state.current_player),
        }
    
    def _get_info(self):
        return {}
    
    def _get_legal_actions_mask(self):
        """Return an array of legal actions for the current player."""
        legal_actions = np.zeros(self.config.num_possible_actions, dtype=np.int8)

        player = self.player_state[self.game_state.current_player]
        num_cards = self.config.cards_per_player

        # Available actions after drawing from deck
        for icard in range(num_cards):
            if player.is_face_down(icard):
                legal_actions[icard] = 1
                legal_actions[icard + num_cards] = 1
        
        # Available actions after picking up from discard pile
        idx = 2 * num_cards
        for n in range(1, self.max_discard() + 1):
            for combo in combinations(np.arange(num_cards), n):
                if all(player.is_face_down(i) for i in combo):
                    legal_actions[idx] = 1
                idx += 1

        return legal_actions

    def _legal_action(self, action_int: int) -> bool:
        legal_actions_mask = self._get_legal_actions_mask()
        return bool(legal_actions_mask[action_int] == 1)
    
    def step(self, action_int: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an action (draw_action, discard_action, flip_action).

        Returns: (observation, reward, done, info)
        """

        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]

        # Decode action int into component actions
        draw_action, discard_action, flip_action = self.decode_action(action_int)

        # Check legality of action
        if self._legal_action(action_int) is False:
            info = {
                'error': f"Illegal action {action_int} for player {self.game_state.current_player}",
                'draw_action': draw_action,
                'discard_action': discard_action,
                'flip_action': flip_action
            }
            return self._get_obs(), -1.0, round_state.complete, True, info

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

        return self._get_obs(), reward, round_state.complete, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reinitialize the environment. If seed is provided, reseed RNGs."""
        # self.rng = random.Random(seed)
        super().reset(seed=seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()
        return (self._get_obs(), {})

    def initialize_game(self):
        self.game_state = GameState(
            current_player=0,
            round_number=0
        )
    
    def initialize_round(self):
        deck = create_standard_deck(self.config.num_decks)
        self.np_random.shuffle(deck)

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
    
    def max_discard(self) -> int:
        player = self.player_state[self.game_state.current_player]
        return min(self.round_state.discard_count, player.num_face_down)

    def encode_actions(
        self, actions: Tuple[Action, Action, Action]
    ) -> int:
        draw_action, discard_action, flip_action = actions

        player = self.player_state[self.game_state.current_player]
        num_cards = player.num_cards

        if draw_action.type == ActionType.DRAW_DECK:
            draw_action_int = 0
            if discard_action.type == ActionType.DISCARD_DRAWN:
                discard_action_int = 0
            elif discard_action.type == ActionType.DISCARD_OWN:
                discard_action_int = num_cards
            flip_action_int = flip_action.data[0]
        elif draw_action.type == ActionType.PICKUP_DISCARD:
            draw_action_int = 2 * num_cards
            idx = 0
            for n in range(1, num_cards + 1):
                for combo in combinations(np.arange(num_cards), n):
                    if discard_action.data == list(combo):
                        discard_action_int = idx
                    idx += 1
            flip_action_int = 0
        return draw_action_int + discard_action_int + flip_action_int
    
    def decode_action(self, action_int: int) -> Tuple[Action, Action, Action]:
        player = self.player_state[self.game_state.current_player]
        num_cards = player.num_cards

        if action_int < 2 * num_cards:
            draw_action = Action(ActionType.DRAW_DECK)
            if action_int < num_cards:
                discard_action = Action(ActionType.DISCARD_DRAWN)
                flip_action = Action(ActionType.FLIP, [action_int])
            else:
                discard_action = Action(ActionType.DISCARD_OWN, [action_int - num_cards])
                flip_action = Action(ActionType.FLIP, [action_int - num_cards])
        else:
            idx = 2 * num_cards
            for n in range(1, num_cards + 1):
                for combo in combinations(np.arange(num_cards), n):
                    if action_int == idx:
                        draw_action = Action(ActionType.PICKUP_DISCARD, n)
                        discard_action = Action(ActionType.DISCARD_OWN, list(combo))
                        flip_action = Action(ActionType.FLIP, list(combo))
                    idx += 1

        return (draw_action, discard_action, flip_action)