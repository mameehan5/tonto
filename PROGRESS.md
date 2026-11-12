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
