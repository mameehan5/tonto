import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Any, Optional

from game.cards import create_standard_deck
from game.state import GameState, RoundState, PlayerState
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
        """
        Apply draw, discard, and flip actions based on action in
        """

        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]

        # Check legality of action
        if self.is_legal_action(act) is False:
            info = {'action': act, 'env': self}
            return self._get_obs(), -1.0, round_state.complete, True, info

        reward = 0.0
        player.previous_known_score = player.known_score

        # Apply action
        action, val, _ = decode_action(act)

        if round_state.phase == Phase.DRAW:
            reward += self._handle_draw(action, val)

        elif round_state.phase == Phase.DISCARD:
            reward += self._handle_discard(action, val)

        elif round_state.phase == Phase.FLIP:
            reward += self._handle_flip(action, val)

        elif round_state.phase == Phase.DECLARE_JOKER:
            reward += self._handle_joker(action, val)

        elif round_state.phase == Phase.END:
            assert len(player.holding_cards) == 0
            assert player.num_to_flip == 0
            assert not np.any(player.visible_undeclared_jokers)
            self.advance_turn()
            if self.round_state.complete:
                # reward += -np.sum(self.compute_player_scores())/10
                reward += -np.sum(self.compute_player_scores())
                    
        return self._get_obs(), reward, self.round_state.complete, False, self._get_info()

    # Handlers ----------------------------------------------------------------

    def _handle_draw(self, act, n=None):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "draw_deck":
            if len(round_state.draw_deck) == 0:
                self.reshuffle()
            cards = [round_state.draw_deck.pop()]
        elif act == "pickup_discard":
            assert n is not None
            cards = [round_state.discard_pile.pop() for _ in range(n)]
        else:
            raise ValueError('Invalid action.')
        player.action = (act, n)
        player.holding_cards = cards
        player.num_to_flip = len(cards)
        round_state.phase = Phase.DISCARD
        return self._compute_reward()
    
    def _handle_discard(self, act, idx=None):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "discard_drawn":
            round_state.discard_pile.append(player.holding_cards.pop())
            round_state.phase = Phase.FLIP
        elif act == "discard_own":
            round_state.discard_pile.append(player.cards[idx])
            player.cards[idx] = player.holding_cards.pop()
            player.flip(idx)
            player.num_to_flip -= 1
            if player.num_to_flip == 0:
                if np.any(player.visible_undeclared_jokers):
                    round_state.phase = Phase.DECLARE_JOKER
                else:
                    round_state.phase = Phase.END
            else:
                round_state.phase = Phase.DISCARD
        else:
            raise ValueError('Invalid action.')
        player.action = (act, idx)
        return self._compute_reward()

    def _handle_flip(self, act, idx):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "flip":
            player.flip(idx)
            player.num_to_flip -= 1
            if np.any(player.visible_undeclared_jokers):
                round_state.phase = Phase.DECLARE_JOKER
            else:
                round_state.phase = Phase.END
        else:
            raise ValueError('Invalid action.')
        player.action = (act, idx)
        return self._compute_reward()
    
    def _handle_joker(self, act, rank):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "declare_joker":
            player.declare_joker(rank)
            if np.any(player.visible_undeclared_jokers):
                round_state.phase = Phase.DECLARE_JOKER
            else:
                round_state.phase = Phase.END
        else:
            raise ValueError('Invalid action.')
        player.action = (act, rank)
        return self._compute_reward()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reinitialize the environment. If seed is provided, reseed RNGs."""
        super().reset(seed=seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()
        return (self._get_obs(), self._get_info())

    def reshuffle(self):
        """Shuffle the discard pile (except the top card) into a new draw deck."""
        rs = self.round_state
        rs.draw_deck = rs.discard_pile[:-1]
        rs.discard_pile = rs.discard_pile[-1:]
        self.np_random.shuffle(rs.draw_deck)

    def initialize_game(self):
        self.game_state = GameState(
            num_players = self.num_players, 
            num_decks = self.num_decks, 
            num_rounds = self.num_rounds, 
            num_cards = self.num_cards
        )
    
    def initialize_round(self):
        deck = create_standard_deck(self.num_decks)
        self.np_random.shuffle(deck)

        self.round_state = RoundState(
            draw_deck=deck,
            discard_pile=[deck.pop()]
        )
        for i in self.active_players:
            self.player_state[i].reset([self.round_state.draw_deck.pop() for _ in range(self.num_cards)])
        if self.game_state.current_player not in self.active_players:
            self.advance_player()
    
    def initialize_players(self):
        self.player_state = [
            PlayerState(i, self.num_cards) for i in range(self.num_players)
        ]
        self.active_players = list(range(self.num_players))

    def advance_round(self):
        tiebreak = self.game_state.current_round >= self.num_rounds
        for i in self.active_players:
            p = self.player_state[i]
            (p.tiebreak_scores if tiebreak else p.individual_scores).append(p.final_flipped_score)
        self.game_state.current_round += 1
        if self.game_state.current_round >= self.num_rounds:
            # Players tied for the lowest score play another round until the tie is broken
            score = lambda i: self.player_state[i].tiebreak_scores[-1] if tiebreak else self.player_state[i].total_score
            low = min(score(i) for i in self.active_players)
            self.active_players = [i for i in self.active_players if score(i) == low]
            if len(self.active_players) == 1:
                self.game_state.complete = True
                self.winner = self.active_players[0]
                return
        self.initialize_round()

    def advance_player(self):
        gs = self.game_state
        gs.current_player = (gs.current_player + 1) % self.num_players
        while gs.current_player not in self.active_players:
            gs.current_player = (gs.current_player + 1) % self.num_players
    
    def advance_turn(self):
        self.advance_player()
        self.round_state.complete = all(self.player_state[i].is_face_up_all() for i in self.active_players)
        player = self.player_state[self.game_state.current_player]
        if player.is_face_up_all():
            self.round_state.phase = Phase.END
        else:
            self.round_state.phase = Phase.DRAW
        
    def _compute_reward(self):
        player = self.player_state[self.game_state.current_player]
        # if player.known_score < player.previous_known_score:
        #     return 0.1
        # else:
        #     return -0.1
        return -0.1*(player.known_score - player.previous_known_score)
        # return -1*(player.known_score - player.previous_known_score)

    def compute_player_scores(self):
        return np.array([self.player_state[i].final_flipped_score for i in self.active_players])

    @property
    def max_draw_from_discard(self):
        player = self.player_state[self.game_state.current_player]
        return min(self.round_state.num_discard, player.num_face_down)
