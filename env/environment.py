import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Any, Optional

from game.engine import TontoGame
from game.actions import Action, compute_action_mask, decode_action, MAX_CARDS, MAX_PLAYERS
from game.phases import Phase


class TontoEnv(gym.Env):
    def __init__(
        self, 
        num_players, 
        num_decks, 
        num_rounds, 
        num_cards, 
        seed: int = 42
    ):
        self.num_players = num_players
        self.num_decks = num_decks
        self.num_rounds = num_rounds
        self.num_cards = num_cards

        self.max_num_players = MAX_PLAYERS
        self.max_num_cards = MAX_CARDS
        self.num_ranks = 13
        self.reset(seed=seed)

        # Check if encode/decode are consistent inverses
        # for action_int in range(self.config.num_possible_actions):
        #     actions = self.decode_action(action_int)
        #     reencoded = self.encode_actions(actions)
        #     assert action_int == reencoded, f"Action encoding/decoding mismatch: {action_int} -> {actions} -> {reencoded}"

        self.observation_space = spaces.Dict({
            "hand_ranks": spaces.MultiDiscrete(
                nvec=[14] * self.max_num_cards, 
                dtype=np.int32
            ), 
            "face_up": spaces.MultiBinary((self.max_num_cards,)),
            "seen": spaces.MultiBinary((self.max_num_cards,)),
            "known_score": spaces.Box(
                low=(-5*2*self.max_num_cards), 
                high=(10*2*self.max_num_cards), 
                shape=(1,), 
                dtype=np.int32
            ), 
            "discarded_cards": spaces.MultiDiscrete(
                nvec=[14] * self.max_num_cards, 
                dtype=np.int32
            ), 
            "drawn_cards": spaces.MultiDiscrete(
                nvec=[14] * self.max_num_cards, 
                dtype=np.int32
            ), 
            "phase": spaces.Discrete(
                5
            ), 
            "current_player": spaces.Discrete(
                self.max_num_players
            ),
            "actions_mask": spaces.MultiBinary(
                (
                    int(Action.NUM_POSSIBLE_ACTIONS),
                )
            ),
        })
        self.action_space = spaces.Discrete(int(Action.NUM_POSSIBLE_ACTIONS))

    def _get_obs(self):
        """Return a gym-compatible observation that matches self.observation_space."""

        player = self.player_state[self.game_state.current_player]

        # Hand ranks
        hand_ranks = np.full((self.max_num_cards), self.num_ranks, dtype=np.int32)
        face_up = np.zeros((self.max_num_cards), dtype=np.int8)
        seen = np.zeros((self.max_num_cards), dtype=np.int8)
        for ic, (c,up,se) in enumerate(zip(
            player.cards, player.face_up, player.private_seen
        )):
            if up: 
                hand_ranks[ic] = c.rank
            face_up[ic] = up
            seen[ic] = se

        # Set discarded cards
        discarded_cards = np.full((self.max_num_cards), self.num_ranks, dtype=np.int32)
        for ic, c in enumerate(reversed(self.round_state.discard_pile[-self.num_cards:])):
            discarded_cards[self.max_num_cards-ic-1] = c.rank

        # Set drawn cards
        drawn_cards = np.full((self.max_num_cards), self.num_ranks, dtype=np.int32)
        for ic, c in enumerate(player.holding_cards):
            drawn_cards[ic] = c.rank

        return {
            "hand_ranks": hand_ranks, 
            "face_up": face_up, 
            "seen": seen, 
            "discarded_cards": discarded_cards, 
            "known_score": np.array([player.known_score], np.int32), 
            "actions_mask": compute_action_mask(self), 
            "phase": int(self.round_state.phase), 
            "drawn_cards": drawn_cards, 
            "current_player": int(self.game_state.current_player), 
        }
    
    def _get_info(self):
        # return {"legal_actions_mask": self._get_legal_actions_mask()}
        return {}
    
    def action_masks(self):
        return compute_action_mask(self).astype(bool)
    
    def is_legal_action(self, action: int):
        action_mask = compute_action_mask(self)
        return bool(action_mask[action])
    
    def step(self, act: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        game = self.game
        if not game.is_legal(act):
            return self._get_obs(), -1.0, game.round_state.complete, True, {'action': act, 'env': self}

        player = game.current
        phase = game.phase
        player.previous_known_score = player.known_score
        game.step(act)

        if phase == Phase.END:
            reward = -float(np.sum(self.compute_player_scores())) if game.round_state.complete else 0.0
        else:
            reward = -0.1 * (player.known_score - player.previous_known_score)
        return self._get_obs(), reward, game.round_state.complete, False, self._get_info()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game = TontoGame(self.num_players, self.num_decks, self.num_rounds, self.num_cards, seed=self.np_random)
        return self._get_obs(), self._get_info()

    # Delegate state to the engine (used by compute_action_mask and the terminal UI)
    game_state = property(lambda self: self.game.game_state)
    round_state = property(lambda self: self.game.round_state)
    player_state = property(lambda self: self.game.player_state)
    active_players = property(lambda self: self.game.active_players)
    winner = property(lambda self: self.game.winner)
    max_draw_from_discard = property(lambda self: self.game.max_draw_from_discard)

    def advance_round(self):
        self.game.next_round()

    def compute_player_scores(self):
        return np.array([self.player_state[i].final_flipped_score for i in self.active_players])
