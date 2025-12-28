from collections import Counter
# from core.cards import Rank

def compute_player_score(cards):
    # Count cards by rank (face) only, not by suit. `Card` is (rank, suit).
    card_counts = Counter(rank for rank, _ in cards)

    score = 0
    for rank, count in card_counts.items():
        # Jokers
        if rank == "JOKER":
            continue  # Jokers are neutral once called

        # Kings
        if rank == 'K':
            # singles & pairs of kings are 0
            if count == 3:
                score -= 10
            elif count >= 4:
                score -= 5 * count
            continue

        # Fives (special handling)
        if rank == '5':
            score -= 5 * count
            continue

        # Face value
        if rank in ('J', 'Q'):
            face_value = 10
        elif rank == 'A':
            face_value = 1
        else:
            face_value = int(rank)

        # General grouping rules
        if count == 1:
            score += face_value
        elif count == 2:
            score += 0
        elif count == 3:
            score -= face_value
        elif count >= 4:
            score -= 5 * count

    return score
