from collections import Counter

def compute_player_score(player, style='flipped', assert_flipped=False):
    if style == 'flipped':
        if assert_flipped is True:
            assert player.is_face_up_all()
        card_counts = Counter(c.rank for c, up in zip(player.cards, player.face_up) if up)
    elif style == 'known':
        card_counts = Counter(c.rank for c, up, seen in zip(player.cards, player.face_up, player.private_seen) if up or seen)
    else:
        raise ValueError(f'Unknown scoring style: {style}')
    return sum(calculate_rank_score(rank, count) for rank, count in card_counts.items())

def calculate_rank_score(rank, count):
    assert 0 <= rank <= 12
    assert count >= 1

    # Kings
    if rank == 12:
        if count == 1 or count == 2:
            return 0
        if count == 3:
            return -10
        elif count >= 4:
            return -5 * count

    # Fives
    if rank == 4:
        return -5 * count

    # Face value
    if rank in [10,11]:
        face_value = 10
    elif rank == 0:
        face_value = 1
    else:
        face_value = int(rank + 1)

    # General grouping rules
    if count == 1:
        return face_value
    elif count == 2:
        return 0
    elif count == 3:
        return -face_value
    elif count >= 4:
        return -5 * count

    raise ValueError(f'No score determined. Rank: {rank}. Count: {count}')

