# Tonto-Inspired Card Game

This repository contains a Python-based implementation of a custom multiplayer card game, playable in the terminal and ready for AI/ML development. The game supports multiple rounds, scoring, and fully enforces the game rules.

---

## Table of Contents

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

## Game Setup

- **Players:** 2–12  
- **Decks:** 1–4 standard decks (52 cards + 2 jokers each)  
- **Cards per player:** 2–20 (same for all players per round)  
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
| Three of same card  | -face value (3 Jacks = -10) |
| Four of a kind      | -20                         |
| Five or more same   | -5 × number of identical    |
| Jokers              | Wildcard (must be called)   |

---

## Drawing Cards

- **From deck:**
  - Draw **1 card**.  
  - May flip the drawn card or **one own unflipped card**.  

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
- Ties are resolved **by the number of rounds played**.

---

## Rounds and Winning

- Number of rounds = 1–10 (set at game start).  
- Scores accumulate across rounds.  
- Player with **lowest cumulative score** at game end wins.

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
