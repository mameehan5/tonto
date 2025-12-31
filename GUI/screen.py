import pygame

#here I just want to draw the screen
class Screen:
    def __init__(self,window):
        pygame.init()
        #I might want to modify this later
        self.window = window 
        #I would pass this to the constructor: pygame.display.set_mode((1500, 900))

        #CLOCK = pygame.time.Clock()

    @property
    def get_screen(self):
        return self.window 