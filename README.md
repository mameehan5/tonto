# Tonto-Inspired Card Game

This repository contains a Python-based implementation of a custom multiplayer card game, playable in the terminal and ready for AI/ML development. The game supports multiple rounds, scoring, and fully enforces the game rules.

---

## Table of Contents

0. [Installation](#installation)
1. [Game Setup](#game-setup)  
2. [Deck and Cards](#deck-and-cards)  
3. [Gameplay Flow](#gameplay-flow)  
4. [Card Values](#card-values)  
5. [Drawing Cards](#drawing-cards)  
6. [Flipping Cards](#flipping-cards)  
7. [Discarding Cards](#discarding-cards)  
8. [Special Rules](#special-rules)  
9. [Scoring](#scoring)  
10. [Rounds and Winning](#rounds-and-winning)  
11. [Terminal UI Notes](#terminal-ui-notes)

---

## Installation

After installing conda, please install the following packages
conda install --yes numpy matplotlib gymnasium
conda install --yes -c conda-forge sb3-contrib

## Play With Friends (web app)

```bash
pip install -r requirements-server.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://<this computer's IP>:8000` on any phone on the same Wi-Fi. Start a table, share the 4-letter code (or the invite link), and fill empty seats with easy, medium or hard bots (hard = search bot). On iPhone, Safari → Share → **Add to Home Screen** makes it open like an app.

- **Friends outside your Wi-Fi:** expose the server with a tunnel, e.g. `cloudflared tunnel --url http://localhost:8000` (gives a public https link), or Tailscale, or deploy the `Dockerfile` to any container host.
- The server holds the real game; each phone only receives its own view (no peeking at others' cards). Games are saved to `rooms/` and survive restarts; reloading the page rejoins your seat.
- Test: `python tests/test_server.py`.

## Quick Start

```bash
python gui/server.py                                # web GUI at http://localhost:8000
python scripts/eval_bots.py greedy random --games 500   # bot win rates (seats shuffled, 95% CI)
python scripts/eval_bots.py ppo:models/best_model.zip greedy greedy   # trained model vs baselines
python tests/simulate_rules.py                      # random-play rule checks
```

The GUI lets you play against bots, step through bot moves, undo, and toggle **Reveal all hands** to see every player's face-down cards (dashed = face-down, 👁 = peeked by owner) for evaluating bot play.

## Training an AI Player

```bash
# 1) Imitation warm start: copy GreedyBot (~2 min)
python training/imitate.py --players 2 --cards 4 --games 20000 --out models/bc_2p4c/bc.zip
# 2) PPO fine-tuning vs greedy/random bots + frozen snapshots of itself
python train_selfplay.py --players 2 --cards 4 --resume models/bc_2p4c/bc.zip --lr 5e-5 --clip 0.1 --out models/my_run
#    (or train from scratch: drop --resume)
python scripts/compare.py random greedy "PPO=ppo:models/my_run/best.zip"   # head-to-head heatmap
```

Each run writes `best.zip`, `eval.csv`, and `eval.png` (win rate vs greedy and random over training).
The training env (`env/selfplay.py`) gives the agent one random seat, and its observation includes its peeked cards and opponents' face-up cards.

### Search bot and expert iteration

`search:N` is a determinized Monte Carlo search bot: for each move it samples the hidden cards N times, plays every candidate move out with greedy, and picks the best average. It beats greedy (2p/8 cards: 55% at N=16, 62% at N=48) at ~0.25 s/move, and is available as a GUI seat and in `eval_bots.py`.

```bash
python training/expert_data.py --cards 8 --games 2000 --samples 32 --workers 8 --out data/exit1   # restartable
python training/train_apprentice.py data/exit1 --init models/bc_2p8c/bc.zip --delta 0.3 --greedy-games 6000 --out models/exit1/apprentice.zip
```

## Code Layout

- `game/engine.py` — `TontoGame`: the rules engine (legal actions, `step`, per-player `view`, `to_dict`/`from_dict`). Everything else wraps this.
- `game/search.py` — `SearchBot` (Monte Carlo search over sampled hidden cards).
- `game/bots.py` — `RandomBot`, `GreedyBot` (decide from their own view only); `make_bot("greedy" | "random" | "ppo:<path>")`.
- `training/policy_bot.py` — wraps a trained MaskablePPO model as a bot. Models in `models/` appear as GUI seat options.
- `env/selfplay.py` — self-play training env (learner vs bots/snapshots, competitive reward). `env/environment.py` — original Gym wrapper.
- `train_selfplay.py`, `training/imitate.py` — PPO training and behavior cloning. `scripts/compare.py`, `scripts/plot_runs.py` — evaluation plots.
- `server/` — multiplayer web app (FastAPI + WebSockets; `server/static/index.html` is the phone client).
- `gui/` — local evaluation GUI (reveals all hands). `ui/terminal.py` — terminal UI.

## Game Setup

- **Players:** 2–12  
- **Decks:** 1–4 standard decks (52 cards + 2 jokers each)  
- **Cards per player:** 2–20, even (same for all players per round)  
- Players × cards must leave at least 2 cards undealt. If players × cards is more than about half the total cards, a warning is shown that the draw deck may run out and reshuffling may be required.  
- At the start of each round, each player privately sees **half of their own hand** (the first half). The other half stays unseen until flipped.  
- **Rounds per game:** 1–10  
- Lead player proposes the game configuration (number of decks, cards, and rounds), and all other players agree.

---

## Deck and Cards

- Standard suits: ♠ ♥ ♦ ♣  
- Ranks: Ace (1), 2–10, Jack, Queen, King  
- Jokers are wildcards and must be **called when flipped**.  
- Cards are visually displayed in the terminal; suits are visible.  

---

## Gameplay Flow

Each round consists of repeating the following **phases per player**, in counterclockwise order:

1. **Draw Phase**  
2. **Flip Phase**  
3. **Discard Phase**

Players continue until **all cards in all hands are face-up**.

---

## Card Values

| Card                | Value                       |
|---------------------|-----------------------------|
| Ace                 | 1                           |
| 2–10                | Face value                  |
| 5                   | -5 (per single 5)           |
| Jack, Queen         | 10                          |
| King                | 0                           |
| Two of same card    | 0 (except two 5s = -10)     |
| Three of same card  | -face value (3 Jacks = -10, 3 Kings = -10, 3 fives = -15) |
| Four of a kind      | -20                         |
| Five or more same   | -5 × number of identical    |
| Jokers              | Wildcard (must be called)   |

---

## Drawing Cards

- **From deck:**
  - Draw **1 card**.  
  - Either keep the drawn card (it replaces a discarded unflipped card and **must be flipped face-up**), or discard it and flip **one own unflipped card**.  

- **From discard pile:**
  - Can draw **1 or more consecutive cards**.  
  - Must flip **all drawn cards**.  
  - Cannot flip own cards during this turn.  
  - Maximum number drawn = min(cards in discard pile, unflipped cards in hand).  

---

## Flipping Cards

- Players may **flip one own card** if they drew from the deck.  
- Flipping **all drawn cards** is mandatory if picked from discard pile.  
- Flipped cards are **publicly visible**.  
- Flipped cards **cannot be discarded**.

---

## Discarding Cards

- **Deck draw:** discard **either** the drawn card **or** one unflipped card from hand.  
- **Discard pile draw:** discard **exactly the same number** of cards as drawn from your hand.  
- Cannot discard flipped cards.  
- Terminal shows **top N cards of the discard pile**, where N = number of cards in your hand.

---

## Special Rules

- If the **draw deck runs out**, shuffle the discard pile **except the top card**.  
- Face-up cards are visible to all players; unflipped cards are private until flipped or discarded.  
- Jokers must be **called immediately when flipped**.

---

## Scoring

- At round end, sum card values for each player.  
- Player with **lowest total score wins the round**.  
- Scores accumulate across rounds (see below).

---

## Rounds and Winning

- Number of rounds = 1–10 (set at game start).  
- Scores accumulate across rounds.  
- Player with **lowest cumulative score** at game end wins.  
- **Ties:** if two or more players tie for the lowest score, only those players play another round. This repeats until the tie is broken.

---

## Terminal UI Notes

- Displays:
  - Player hand (hidden/unflipped cards shown as 🂠)  
  - Top discard pile (dynamic: top N cards, N = hand size)  
  - Drawn card(s)  
  - Flipped cards for current turn  
- Actions are listed and player selects via number input.  
- Supports undo/rewind for debugging and replay.

---

## ML / AI Notes

- Only **public and privately known info** is available to computer players.  
- Terminal game is fully compatible with high-speed simulation for AI training.

---
