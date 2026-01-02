import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Any, Optional
from itertools import combinations

from math import comb
import copy

from game.cards import create_standard_deck
from game.state import GameState, RoundState, PlayerState 
from game.scoring import compute_player_score
from core.actions import apply_draw_action, apply_discard_action, apply_flip_action
from rl.actions import Action, compute_action_mask, decode_action



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
        self.reset(seed=seed)

        # Check if encode/decode are consistent inverses
        # for action_int in range(self.config.num_possible_actions):
        #     actions = self.decode_action(action_int)
        #     reencoded = self.encode_actions(actions)
        #     assert action_int == reencoded, f"Action encoding/decoding mismatch: {action_int} -> {actions} -> {reencoded}"

        num_ranks = 13
        self.observation_space = spaces.Dict({
            "hand_rank": spaces.Box(
                low=-1, 
                high=12, 
                shape=(self.game_state.max_num_cards,), 
                dtype=np.int8
            ), 
            # "player_card_ranks": spaces.MultiBinary(
            #     (
            #         self.num_players, 
            #         self.num_cards, 
            #         num_ranks + 1 # +1 for face-down indicator
            #     )
            # ),
            # "player_known_scores": spaces.Box(
            #     low=self.config.min_possible_score, 
            #     high=self.config.max_possible_score,
            #     shape=(self.num_players,), 
            #     dtype=np.int32
            # ),
            # TODO: this may be better to track counts of card ranks of unflipped
            # "num_unflipped_cards": spaces.Box(
            #     low=0, 
            #     high=self.config.num_total_cards,
            #     shape=(1,), 
            #     dtype=np.int32, 
            # ),
            "face_up": spaces.MultiBinary((self.game_state.max_num_cards,)),
            "seen": spaces.MultiBinary((self.game_state.max_num_cards,)),
            
            # NEED TO ADD BELOW
            # "discarded_cards": spaces.MultiBinary(
            #     (
            #         self.num_cards, 
            #         num_ranks + 1 # +1 for empty slots
            #     )
            # ),


            # "current_player": spaces.Discrete(
            #     self.num_players
            # ),
            # "num_cards_per_player_half": spaces.Discrete(
            #     self.config.max_num_cards // 2
            # ),


            # self.observation_space = spaces.Dict({
            #     ,
                
            #     "is_joker": spaces.MultiBinary(MAX_CARDS),
            #     "joker_declared": spaces.MultiBinary(MAX_CARDS),
            #     "phase": spaces.Discrete(len(PHASES)),
            # })

            # "legal_actions_mask": spaces.MultiBinary(
            #     (
            #         self.config.num_possible_actions,
            #     )
            # ),
            # Need to add top of card pile filtered on legal actions
        })
        self.action_space = spaces.Discrete(int(Action.NUM_POSSIBLE_ACTIONS))

    def _get_obs(self):
        """Return a gym-compatible observation that matches self.observation_space."""

        num_players = self.num_players
        num_cards = self.num_cards
        num_ranks = 13

        player = self.player_state[self.game_state.current_player]

        # Player card ranks
        # player_card_ranks = np.zeros((num_players, num_cards, num_ranks + 1), dtype=np.int8)
        # for ip, player in enumerate(self.player_state):
        #     for ic, (c,up) in enumerate(zip(player.cards, player.face_up)):
        #         if up is True:
        #             player_card_ranks[ip, ic, c.rank] = 1
        #         else:
        #             player_card_ranks[ip, ic, num_ranks] = 1  # face-down indicator

        # Hand ranks
        hand_ranks = np.zeros((self.game_state.max_num_cards), dtype=np.int8)
        face_up = np.zeros((self.game_state.max_num_cards), dtype=np.int8)
        seen = np.zeros((self.game_state.max_num_cards), dtype=np.int8)
        for ic, (c,up,se) in enumerate(zip(
            player.cards, player.face_up, player.private_seen
        )):
            hand_ranks[ic] = c.rank
            face_up[ic] = up
            seen[ic] = se


        # Player known score
        # player_known_scores = np.empty((self.num_players), dtype=np.int32)
        # for ip, player in enumerate(self.player_state):
        #     if ip == self.game_state.current_player:
        #         player_known_scores[ip] = compute_player_score(player, style='known')
        #     else:
        #         player_known_scores[ip] = compute_player_score(player, style='flipped')
            
        # Calculate unknown card size
        # num_unflipped_cards = len(self.round_state.draw_deck)
        # for player in self.player_state:
        #     num_unflipped_cards += player.num_face_down

        # Set discarded cards
        # discarded_cards = np.zeros((num_cards, num_ranks + 1), dtype=np.int8)
        # discarded_cards[:, num_ranks] = 1
        # for ic, c in enumerate(reversed(self.round_state.discard_pile[-num_players:])):
        #     discarded_cards[-ic, c.rank] = 1
        #     discarded_cards[-ic, num_ranks] = 0

        return {
            "hand_ranks": hand_ranks, 
            "face_up": face_up, 
            "seed": seen
            # "player_card_ranks": player_card_ranks,
            # "player_known_scores": player_known_scores, 
            # # "num_unflipped_cards": np.array([num_unflipped_cards], dtype=np.int32), 
            # "discarded_cards": discarded_cards, 
            # "current_player": self.game_state.current_player,
            # "num_cards_per_player_half": self.num_players // 2,
            # "legal_actions_mask": self._get_legal_actions_mask()
        }
    
    def _get_info(self):
        # return {"legal_actions_mask": self._get_legal_actions_mask()}
        return {}
    
    # def _get_legal_actions_mask(self):
    #     """Return an array of legal actions for the current player."""
    #     legal_actions = np.zeros(Action.NUM_POSSIBLE_ACTIONS, dtype=np.int8)

    #     player = self.player_state[self.game_state.current_player]
    #     num_cards = self.num_cards

    #     # If all cards are flipped, we must pass
    #     if player.is_face_up_all():
    #         legal_actions[0] = 1

    #     # Available actions after drawing from deck
    #     for icard in range(num_cards):
    #         if player.is_face_down(icard):
    #             legal_actions[icard + 1] = 1
    #             legal_actions[icard + num_cards + 1] = 1
        
    #     # Available actions after picking up from discard pile
    #     idx = 2 * num_cards + 1
    #     for n in range(1, self.max_discard() + 1):
    #         for combo in combinations(np.arange(num_cards), n):
    #             if all(player.is_face_down(i) for i in combo):
    #                 legal_actions[idx] = 1
    #             idx += 1

    #     return legal_actions

    def is_legal_action(self, action: int):
        action_mask = compute_action_mask(self)
        return action_mask[action]
    
    def step(self, act: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Apply draw, discard, and flip actions based on action in
        """

        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]

        # Check legality of action

        if self.is_legal_action(act) is False:
            # draw_action, discard_action, flip_action = self.decode_action(action)
            # info = {
            #     'error': f"Illegal action {action} for player {self.game_state.current_player}",
            #     'draw_action': draw_action,
            #     'discard_action': discard_action,
            #     'flip_action': flip_action
            # }
            info = {}
            return self._get_obs(), -1.0, round_state.complete, True, info

        # Calculate best and worst possible outcome using available legal actions
        # best, worst = self.possible_scores_extrema()

        # Apply action
        # self.apply_action(action)

        reward = 0.0

        action, val, _ = decode_action(act)

        if round_state.phase == "draw":
            reward += self._handle_draw(action, val)

        elif round_state.phase == "discard":
            reward += self._handle_discard(action, val)

        elif round_state.phase == "flip":
            reward += self._handle_flip(action, val)

        if round_state.phase in ["discard", "flip"]:
            if player.num_to_flip == 0:
                done = True
                assert len(player.holding_cards) == 0
        
        # if round_state.phase == 'end_'
        # TODO: call jokers at the end
        # oh wait, we need to only do single action on one step, calling joker
        # is separate action

        # if ():
            # reward += self._round_reward()
            # done = True

        # Compute reward based on score compared to best and worst possibilities
        # score = compute_player_score(player, style='known')
        # reward = self.compute_reward(score, best, worst)
        # reward = 1.0

        # Advance turn
        
        return self._get_obs(), reward, round_state.complete, False, self._get_info()

    def apply_action(self, action_int: int):
        # Perform draw -> discard -> flip using core action helpers
        draw_action, discard_action, flip_action = self.decode_action(action_int)
        drawn_cards = apply_draw_action(self, draw_action)
        if len(drawn_cards) > 0:
            apply_discard_action(self, drawn_cards, discard_action)
            apply_flip_action(self, flip_action)

    # Handlers ----------------------------------------------------------------

    def _handle_draw(self, act, n=None):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "draw_deck":
            cards = [round_state.draw_deck.pop()]
        elif act == "pickup_discard":
            assert n is not None
            cards = [round_state.discard_pile.pop() for _ in range(n)]
        else:
            raise ValueError('Invalid action.')
        player.action = (act, n)
        player.holding_cards = cards
        player.num_to_flip = len(cards)
        round_state.phase = "discard"
        return 0.0
    
    def _handle_discard(self, act, idx=None):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "discard_drawn":
            round_state.discard_pile.append(player.holding_cards.pop())
            round_state.phase = "flip"
        elif act == "discard_own":
            round_state.discard_pile.append(player.cards[idx])
            player.cards[idx] = player.holding_cards.pop()
            player.flip(idx)
            player.num_to_flip -= 1
            round_state.phase = "end_turn" if player.num_to_flip == 0 else "discard"
        else:
            raise ValueError('Invalid action.')
        player.action = (act, idx)
        return 0.0

    def _handle_flip(self, act, idx):
        round_state = self.round_state
        player = self.player_state[self.game_state.current_player]
        if act == "flip":
            player.flip(idx)
            round_state.phase = "end_turn"
        else:
            raise ValueError('Invalid action.')
        player.action = (act, idx)
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reinitialize the environment. If seed is provided, reseed RNGs."""
        super().reset(seed=seed)
        self.initialize_game()
        self.initialize_players()
        self.initialize_round()
        return (self._get_obs(), self._get_info())

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
        for player in self.player_state:
            player.reset([self.round_state.draw_deck.pop() for _ in range(self.num_cards)])
    
    def initialize_players(self):
        self.player_state = [
            PlayerState(i, self.num_cards) for i in range(self.num_players)
        ]

    def advance_round(self):
        for player in self.player_state:
            score = compute_player_score(player, style='flipped')
            player.individual_scores.append(score)
        self.game_state.round_number += 1
        if self.game_state.round_number >= self.config.num_rounds:
            self.game_state.complete = True
    
    def advance_turn(self):
        self.game_state.current_player = (self.game_state.current_player + 1) % self.num_players
        self.round_state.complete = all(p.is_face_up_all() for p in self.player_state)
        self.round_state.phase = 'draw'

    @property
    def max_draw_from_discard(self):
        player = self.player_state[self.game_state.current_player]
        return min(self.round_state.num_discard, player.num_face_down)

    def possible_scores_extrema(self):
        best = self.config.max_possible_score
        worst = self.config.min_possible_score
        legal_actions_mask = np.array(self._get_legal_actions_mask(), dtype=bool)
        for action_int, legal in enumerate(legal_actions_mask):
            env_tmp = copy.deepcopy(self)
            if legal == 0:
                continue

            env_tmp.apply_action(action_int)
            score = compute_player_score(env_tmp.player_state[env_tmp.game_state.current_player], style='known')
            best = min(score, best) 
            worst = max(score, worst)
        return best, worst

    def compute_reward(self, score, best, worst):
        if score == best:
            return 1.0
        elif score == worst:
            return 0.0
        reward = (score - worst)/(best - worst)
        assert 0.0 <= reward <= 1.0
        return reward

    def compute_player_scores(self, assert_flipped=False):
        return [
            compute_player_score(p, style='flipped', assert_flipped=assert_flipped) for p in self.player_state
        ]