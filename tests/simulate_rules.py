"""Random-play rule checks. Usage: python tests/simulate_rules.py [games_per_config]"""
import sys, traceback
from collections import Counter
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import json
from env.environment import TontoEnv
from game.engine import TontoGame
from game.actions import compute_action_mask, decode_action, Action
from game.phases import Phase
from game.state import rank_score

issues = Counter()
examples = {}
def flag(k, msg):
    issues[k] += 1
    examples.setdefault(k, msg)

def all_ids(env):
    ids = [c.id for c in env.round_state.draw_deck] + [c.id for c in env.round_state.discard_pile]
    for p in [env.player_state[i] for i in env.active_players]:
        ids += [c.id for c in p.cards] + [c.id for c in p.holding_cards]
    return ids

def ref_score(cards):
    # README table: singles face value (5=-5, K=0), pairs 0 (5s=-10), triples -face, 4 = -20, 5+ = -5n
    fv = {0:1, 4:-5, 10:10, 11:10, 12:0}
    s = 0
    for r, n in Counter(c.rank for c in cards).items():
        v = fv.get(r, r+1)
        if n == 1: s += v
        elif n == 2: s += -10 if r == 4 else 0
        elif n == 3 and r not in (4, 12): s += -v
        elif n == 3 and r == 4: s += -15
        elif n == 3 and r == 12: s += -10
        else: s += -5*n
    return s

def play(seed, P, D, R, N):
    env = TontoEnv(P, D, R, N, seed=seed)
    rng = np.random.default_rng(seed)
    total = D*54
    jokers_seen = 0
    while not env.game_state.complete:
        steps = 0
        while not env.round_state.complete:
            steps += 1
            if steps > 20000:
                flag('round never ends', f'seed={seed}'); return
            p = env.player_state[env.game_state.current_player]
            mask = compute_action_mask(env)
            if not mask.any():
                flag('no legal actions', f'seed={seed} phase={env.round_state.phase}'); return
            if len(set(all_ids(env))) != total or len(all_ids(env)) != total:
                flag('card conservation', f'seed={seed}')
            if env.game_state.current_player not in env.active_players: flag('inactive player acted', '')
            ph = env.round_state.phase
            if ph == Phase.DECLARE_JOKER: jokers_called[0] += 1
            if ph == Phase.DRAW:
                mx = min(len(env.round_state.discard_pile), p.num_face_down)
                n_pick = mask[Action.PICKUP_DISCARD_0:Action.DISCARD_DRAWN].sum()
                if n_pick != mx: flag('discard draw limit', f'{n_pick} vs {mx}')
                if len(env.round_state.discard_pile) == 0: flag('empty discard pile at draw', '')
                if len(env.round_state.draw_deck) == 0: reshuffles[0] += 1
            if ph == Phase.DISCARD:
                for i in np.where(mask[Action.DISCARD_OWN_0:Action.FLIP_0])[0]:
                    if p.face_up[i]: flag('can discard face-up card', '')
            if steps % 37 == 0:
                d = env.game.to_dict()
                env.game = TontoGame.from_dict(json.loads(json.dumps(d)))
                if env.game.to_dict() != json.loads(json.dumps(d)): flag('serialization round-trip', f'seed={seed}')
                for v in range(P):
                    for q in env.game.view(v)['players']:
                        if q['id'] != v and any(c['card'] and not c['face_up'] for c in q['cards']):
                            flag('view leaks hidden card', f'seed={seed}')
            legal = np.where(mask)[0]
            a = int(rng.choice(legal))
            name, val, _ = decode_action(a)
            before_up = p.face_up.copy()
            try:
                env.step(a)
            except Exception as e:
                flag(f'crash: {type(e).__name__}: {e}', f'seed={seed} P={P} D={D} N={N} action={name}')
                return
            # face-up cards never become face-down
            if np.any(before_up & ~p.face_up): flag('card un-flipped', '')
            for c, up in zip(p.cards, p.face_up):
                if up and c.suit == 4: jokers_seen += 1
        # round end checks
        for p in env.player_state:
            if not p.is_face_up_all(): flag('round ended with face-down cards', '')
            if p.player_number in env.active_players and any(c.suit == 4 and c.rank == 13 for c in p.cards):
                flag('joker scored without being called', f'hand={[c.card_str for c in p.cards]} score={p.flipped_score}')
            if p.flipped_score != ref_score(p.cards):
                flag('score mismatch vs README', f'hand={[c.card_str for c in p.cards]} code={p.flipped_score} readme={ref_score(p.cards)}')
        env.advance_round()
    tots = [p.total_score for p in env.player_state]
    w = env.player_state[env.winner]
    if w.total_score != min(tots): flag('winner not lowest total', f'{tots} winner={env.winner}')
    tied = [i for i in range(P) if tots[i] == min(tots)]
    if len(tied) > 1 and not w.tiebreak_scores: flag('tie not replayed', f'{tots}')
    if any(env.player_state[i].tiebreak_scores for i in range(P) if i not in tied): flag('non-tied player in tiebreak', '')
    if env.game_state.current_round > R: tiebreaks[0] += 1
    return
    tot = [p.total_score for p in env.player_state]
    if [sum(p.individual_scores) for p in env.player_state] != tot or any(len(p.individual_scores) != R for p in env.player_state):
        flag('cumulative score wrong', '')

configs = [(12,1,1,4),(12,4,2,16),(10,4,1,20),(3,2,3,20),(2,1,10,2),(2,1,3,4),(4,2,4,8),(3,1,2,6),(4,1,2,12),(2,1,1,2),(6,1,1,8),(4,4,2,12),(8,1,1,6),(5,1,1,10)]
jokers_called=[0]; reshuffles=[0]; tiebreaks=[0]
import warnings; warnings.simplefilter('ignore')
games = 0
for cfg in configs:
    for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 100):
        play(seed, *cfg); games += 1
print('games:', games, '| joker calls:', jokers_called[0], '| games needing tiebreak:', tiebreaks[0], '| reshuffles:', reshuffles[0])
print('PASS' if not issues else 'FAIL')
for k, n in issues.most_common():
    print(f'{n:6d}  {k}\n        e.g. {examples[k]}')
