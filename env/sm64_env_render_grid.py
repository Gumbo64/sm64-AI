import pygame
from PIL import Image

class SM64_ENV_RENDER_GRID:
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, mode="normal"):
        pygame.init()

        self.N_RENDER_COLUMNS = 5
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.mode = mode

        self.RENDER_WINDOW_WIDTH = self.N_RENDER_COLUMNS * IMG_WIDTH + 100
        self.RENDER_WINDOW_HEIGHT = self.N_RENDER_COLUMNS * IMG_HEIGHT + 100
        self.window = pygame.display.set_mode((self.RENDER_WINDOW_WIDTH,self.RENDER_WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")


    def render_game(self, np_imgs):    
        pygame.event.get()
        n_players = np_imgs.shape[0]   
        # print(np_imgs.shape)
        imgs = [0 for i in range(n_players)]
        for i in range(n_players):
            imgs[i] = Image.fromarray(np_imgs[i, :, :, 0], 'L')

        if self.mode == "tag":
            # put hiders and seekers together
            tmp = [0 for i in range(n_players)]
            tmp[::2] = imgs[0:n_players//2] 
            tmp[1::2] = imgs[n_players//2:]
            imgs = tmp



        self.window.fill((0, 0, 0))
        for i in range(len(imgs)):
            # if you get a NULL pointer access, then MAX_PLAYERS doesn't line up with the C code
            tmp = imgs[i].convert("RGB")
            surface = pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode)
            self.window.blit(surface, ((i % self.N_RENDER_COLUMNS) * self.IMG_WIDTH, (i // self.N_RENDER_COLUMNS) * self.IMG_HEIGHT))

        pygame.display.flip()