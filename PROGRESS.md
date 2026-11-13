# Progress Log

## 2026-11-12 — Rule audit + fixes (branch `fix-jokers-reshuffle-limits`)

**Audit:** 1,800 random games vs README rules. Card conservation, discard limits, face-up rules, round end, and scoring were correct.

**Fixed**
- Jokers were never called (deck used rank 13, code checked -1). Now rank 13 = undeclared everywhere; `known_score` joker logic fixed.
- Deck ran out → crash. Now reshuffles discard pile (keeping top card).
- Limits: 2–12 players, 2–20 cards (even), 1–4 decks, 1–10 rounds. Rejects setups leaving <2 cards undealt.
- Warning when 2·players·cards + 1 > total cards (reshuffle likely).
- Ties: tied players replay rounds until broken. Tiebreak scores stored separately (`tiebreak_scores`); `env.winner` set.
- README: half-hand peek, kept drawn card must be flipped, tie rule, limits, 3 Kings = −10, 3 fives = −15.

**Confirmed correct as-is:** 3 fives = −15, 3 Kings = −10, kept deck draw must be flipped face-up.

**Verified:** 2,800 random games (incl. 12 players, 20 cards): no violations. ~10k joker calls, 6,270 reshuffles, 93 tiebreaks.

**Open**
- Trained models in `models/` incompatible (action/obs spaces grew) — retrain.
- Tiebreak rounds: only tied players act; check effect on training rewards.
- `game/rules.py` is dead code (imports nonexistent `GamePhase`).

## 2026-11-12 — Assessment: ML player + multiplayer app

**Training gaps found**
- Obs hides the agent's own peeked cards (`hand_ranks` only shows face-up).
- Obs lacks opponents' face-up cards, deck size, discard depth.
- Round-end reward = −sum of *all* players' scores → cooperative, not competitive. Rewards from different seats mix in one PPO trajectory.
- `train_ppo.py` defaults to 1 player (solitaire).
- Eval reward noisy over 880k steps (−10.6 → −7.6); no baseline to compare against.

**Plan**
1. Refactor: pure `TontoGame` engine (state, legal actions, apply, per-player view, serialize). Gym env and server both wrap it.
2. Baselines: random + greedy heuristic bot; eval = win rate.
3. Training: fix obs/reward, self-play wrapper (learner vs frozen/heuristic opponents), start 2p/4 cards, scale up.
4. Server: FastAPI + WebSockets, rooms by code, server-authoritative hidden info, bots fill seats.
5. Client: web/PWA first (works on iPhone), native SwiftUI later.

## 2026-11-12 — Step 1: engine refactor + GUI (branch `engine-refactor-gui`)

**Done**
- `game/engine.py` `TontoGame`: rules engine with legal actions, `step`, per-player `view` (hides others' face-down cards), JSON `to_dict`/`from_dict`, move log.
- Gym env is now a thin wrapper. Same seeds → identical games as before (rule harness counts match exactly).
- `game/bots.py`: `RandomBot`, `GreedyBot` (one-step greedy on estimated hand score). Greedy beats random ~96% (2p, 4 or 8 cards).
- Web GUI (`python gui/server.py`): reveal all hands, auto/step bots, undo, click cards to act, log, scores.
- `tests/simulate_rules.py` (rule harness, now also checks serialization + view privacy), `scripts/eval_bots.py`.

**Bug fixed:** `known_score` used `face_up & private_seen` (should be `|`), so peeked cards never counted as known. This feeds PPO reward shaping.

**Next:** Step 2/3 — training fixes (obs, competitive reward, self-play vs greedy), eval by win rate vs greedy.

## 2026-11-12 — Step 2: baselines + evaluation (branch `baselines-eval`)

**Done**
- `scripts/eval_bots.py`: any mix of bots, seats shuffled each game (rotation caused a turn-order bias), win rate ± 95% CI.
- `make_bot(spec)`: `greedy`, `greedy_v0`, `random`, `ppo:<path>`. `PolicyBot` runs trained models; GUI lists `models/*.zip` as seats.
- `make_obs(game)` builds the PPO observation from any game state (env uses it too).
- Tuned `GreedyBot` (sweep over pickup threshold × unknown-card value): now threshold 5, unknown 2.5. Old settings kept as `greedy_v0`.
- Old models in `models/` fail to load (obs/action spaces changed) — retrain in step 3.

**Baselines** (1 round, 1 deck, 1000 games)
| Matchup | Cards | Win rate |
|---|---|---|
| greedy vs random | 4 | 95.4% |
| greedy vs random | 8 | 98.4% |
| greedy vs greedy_v0 | 4 | 52.9% |
| greedy vs greedy_v0 | 8 | 67.1% |
| greedy vs 3× greedy_v0 | 6 | 36.0% (even = 25%) |

**Target for step 3:** PPO beats `greedy` > 50% head-to-head (2p, 4 cards), then larger configs.

## 2026-11-12 — Step 3: training fixes + first models (branch `selfplay-training`)

**Built**
- `env/selfplay.py`: learner takes a random seat, opponents are bots/frozen snapshots. Obs now includes own peeked cards, opponents' face-up cards, discard top, deck size. Competitive terminal reward + potential-based shaping. Forced moves auto-applied.
- `train_selfplay.py`: MaskablePPO, eval vs greedy/random every N steps, `eval.png` curve, snapshot pool, `--resume`.
- `training/imitate.py`: behavior cloning from GreedyBot (99.5% move agreement, 2 min).
- `scripts/compare.py` (head-to-head heatmap), `scripts/plot_runs.py`.

**Results** (2 players; win rate vs tuned greedy, 1000 fixed deals) — see `plots/training_runs.png`, `plots/compare_2p4c.png`
| Run | Setup | Best vs greedy |
|---|---|---|
| A | scratch, win reward, 4 cards | 42.8% (plateau by 300k) |
| B | scratch, own-score reward, 4 cards | 43.0% (faster start) |
| C | imitation + PPO, score reward, 4 cards | ~50%, mean score 0.63 → 0.07 |
| D | C + win reward | ~50% |
| E | imitation + PPO, 8 cards | ~48% |

**Takeaways**
- PPO from scratch reaches ~43% vs greedy, 95% vs random. Gap spread across all phases (diagnosed by swapping in greedy per phase).
- Imitation warm start → greedy-level immediately; PPO fine-tuning is stable but hasn't exceeded greedy.
- 4 cards leaves little room for skill (greedy vs greedy_v0 only 53%).

**Open / next ideas**
- Search-based play (sample hidden cards, roll out with greedy) → likely beats greedy; could then be distilled into the network (expert iteration).
- Larger tables (4p/6 cards) where reading opponents matters more; longer runs, more envs (only 2 CPUs here).
- Joker calls: PPO almost always calls 5 (rare event, little signal).

## 2026-11-12 — Expert iteration, round 1 (branch `expert-iteration`)

**Built**
- `game/search.py` `SearchBot` (`search:N`): sample hidden cards N times, play each candidate move out with greedy, pick best average (win/loss + small margin). Same sampled deal for every move.
- Speedups: GreedyBot reads state directly (4× faster, identical moves over 300 games); `TontoGame.clone()` 31 µs (12× faster than deepcopy); logging off in rollouts.
- `training/expert_data.py` (restartable search self-play data), `training/train_apprentice.py` (fit network to search moves; `--delta` keeps greedy's move unless search is clearly better; `--greedy-games` mixes in greedy data; `--soft` value-based targets), `training/obs_greedy.py` (greedy's move from an obs; matches 9,416/9,416).
- PolicyBot uses greedy for joker calls when `greedy_jokers` is set (search data has no joker decisions).
- GUI seats: `search:16`, `search:48`.

**Results** (2 players, 8 cards, vs tuned greedy)
| Bot | Win rate |
|---|---|
| search:16 (500 games) | **54.6% ± 4.4%** |
| search:48 (300 games) | **62.3% ± 5.5%** |
| apprentice, raw search labels (35k moves) | 37% |
| apprentice, delta 0.3 | 42% |
| apprentice, delta 0.3 + 217k greedy moves | 48.6% |

Search speed: ~230–340 ms/move median (2p/8, 4p/6, 4p/8), max ~1–2 s → fine as an app opponent.

**Takeaways**
- Search is the first bot that clearly beats greedy, with no neural network.
- Distilling it into the network failed at this data scale: ~40k searched moves (2 CPUs, ~1 h). Search choices are noisy (greedy agrees with search on 58%), and the network can't separate real improvements from noise.
- Next: generate 300k+ searched moves on a many-core machine (same script, `--workers`), more samples per move (48+) for cleaner labels, then retrain; or use search directly as the app's computer player.
