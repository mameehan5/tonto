from collections import Counter

def calculate_rank_score(rank, count):
    score = 0
    
    # Jokers
    if rank == "JOKER":
        return score  # Jokers are neutral once called

    # Kings
    if rank == 'K':
        # singles & pairs of kings are 0
        if count == 3:
            score -= 10
        elif count >= 4:
            score -= 5 * count
        return score

    # Fives (special handling)
    if rank == '5':
        score -= 5 * count
        return score

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

def compute_player_score(player, style='flipped'):
    if style == 'flipped':
        card_counts_flipped = Counter(rank for (rank, _, _), up in zip(player.cards, player.face_up) if up)
        score = sum(calculate_rank_score(rank, count) for rank, count in card_counts_flipped.items())
    elif style == 'known':
        card_counts_known = Counter(rank for (rank, _, _), up, seen in zip(player.cards, player.face_up, player.private_seen) if up or seen)
        score = sum(calculate_rank_score(rank, count) for rank, count in card_counts_known.items())
    else:
        raise ValueError(f'Unknown scoring style: {style}')
    return score
