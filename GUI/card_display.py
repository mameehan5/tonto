import pygame
import sys

# ---------- Constants ----------
CARD_WIDTH = 120
CARD_HEIGHT = 180
CARD_RADIUS = 12

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 0, 0)
GREEN = (0, 120, 0)
FONT_PATH = "DejaVuSans.ttf"

# ---------- Card drawing ----------
def draw_card(
    surface,
    x,
    y,
    rank="A",
    suit="♠"
):
    # Card color
    color = RED if suit in ["♥", "♦"] else BLACK

    card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)

    # Card background
    pygame.draw.rect(
        surface,
        WHITE,
        card_rect,
        border_radius=CARD_RADIUS
    )

    # Card border
    pygame.draw.rect(
        surface,
        BLACK,
        card_rect,
        width=2,
        border_radius=CARD_RADIUS
    )

    # Fonts
    rank_font = pygame.font.Font(FONT_PATH, 36)
    suit_font = pygame.font.Font(FONT_PATH, 32)
    center_font = pygame.font.Font(FONT_PATH, 64)

    # Top-left rank & suit
    rank_surf = rank_font.render(rank, True, color)
    suit_surf = suit_font.render(suit, True, color)

    surface.blit(rank_surf, (x + 8, y + 6))
    surface.blit(suit_surf, (x + 8, y + 36))

    # Bottom-right (rotated)
    rank_surf_rot = pygame.transform.rotate(rank_surf, 180)
    suit_surf_rot = pygame.transform.rotate(suit_surf, 180)

    surface.blit(
        rank_surf_rot,
        (x + CARD_WIDTH - rank_surf_rot.get_width() - 8,
         y + CARD_HEIGHT - rank_surf_rot.get_height() - 36)
    )
    surface.blit(
        suit_surf_rot,
        (x + CARD_WIDTH - suit_surf_rot.get_width() - 8,
         y + CARD_HEIGHT - suit_surf_rot.get_height() - 6)
    )

    # Center suit
    center_surf = center_font.render(suit, True, color)
    center_rect = center_surf.get_rect(center=card_rect.center)
    surface.blit(center_surf, center_rect)


