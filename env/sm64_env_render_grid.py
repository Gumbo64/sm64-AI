import pygame
from PIL import Image
import numpy as np

class SM64_ENV_RENDER_GRID:
    def __init__(self, IMG_WIDTH, IMG_HEIGHT,N_RENDER_COLUMNS=5, N_RENDER_ROWS=6, mode="normal", coloured=False):
        pygame.init()
        self.coloured = coloured
        self.N_RENDER_COLUMNS = N_RENDER_COLUMNS
        self.N_RENDER_ROWS = N_RENDER_ROWS
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.mode = mode

        self.RENDER_WINDOW_WIDTH = self.N_RENDER_COLUMNS * IMG_WIDTH
        self.RENDER_WINDOW_HEIGHT = self.N_RENDER_ROWS * IMG_HEIGHT
        self.window = pygame.display.set_mode((self.RENDER_WINDOW_WIDTH,self.RENDER_WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")

    def render_game(self, observations):  
        pygame.event.get()
        # Pettingzoo gives a dictionary, otherwise it's just a numpy array
        imgs = []

        # mixed data types in the observations becomes one
        if type(observations) == tuple:
            observations = observations[0]
            # print(observations.shape)
        
        if type(observations) == dict:
            keys = list(observations.keys())
            imgs = [0 for i in range(len(keys))]
            for i in range(len(keys)):
                # the 0 is the frame_stack dimension, we only want one frame though
                imgs[i] = Image.fromarray(observations[keys[i]][:,:,0].astype(np.uint8), 'L')
        else:
            imgs = [0 for i in range(len(observations))]
            n_players = observations.shape[0]
            # print(np_imgs.shape)

            for i in range(n_players):
                if self.coloured:
                    imgs[i] = Image.fromarray(observations[i].astype(np.uint8), 'RGB')
                else:
                    imgs[i] = Image.fromarray(observations[i, :, :, 0].astype(np.uint8), 'L')

        self.window.fill((0, 0, 0))
        for i in range(len(imgs)):
            # if you get a NULL pointer access, then MAX_PLAYERS doesn't line up with the C code
            tmp = imgs[i].convert("RGB")
            surface = pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode)
            self.window.blit(surface, ((i % self.N_RENDER_COLUMNS) * self.IMG_WIDTH, (i // self.N_RENDER_COLUMNS) * self.IMG_HEIGHT))

        pygame.display.flip()