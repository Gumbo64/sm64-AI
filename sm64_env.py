from pettingzoo import ParallelEnv
import ctypes
import pygame
import random
from PIL import Image
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import functools

class gfxPixels(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ), 
        ("height",ctypes.c_int ), 
        ("width",ctypes.c_int ), 
    ]

class inputStruct(ctypes.Structure):
    _fields_ = [
        ("stickX",ctypes.c_short), 
        ("stickY",ctypes.c_short ), 
        ("buttonInput",ctypes.c_bool * 3), # goes A,B,Z
    ]

class SM64_ENV(ParallelEnv):
    metadata = {
        "name": "sm64",
    }

    def __init__(self):
        self.action_spaces = {
            "mario": Discrete(12),
        }


        self.MAX_PLAYERS = 4
        self.N_SCREENS_WIDTH = 5
        self.WINDOW_WIDTH = 256 * self.N_SCREENS_WIDTH
        self.WINDOW_HEIGHT = 144 * self.MAX_PLAYERS // self.N_SCREENS_WIDTH + 1
        self.imgs = list(range(self.MAX_PLAYERS))

        pygame.init()
        self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")

        self.dll = ctypes.CDLL(r"./build/us_pc/sm64.dll")
        self.dll.step_pixels.argtypes = [inputStruct * self.MAX_PLAYERS]
        self.dll.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(gfxPixels))
        
        self.dll.main_func()
        for i in range(10):
            self.step({})
        print("making marios")
        self.dll.makemariolol()
        self.reset()

    def reset(self, seed=None, options=None):
        self.dll.reset()

    def step(self, actions):
        inputstructs = (inputStruct * self.MAX_PLAYERS)()
        for i in range(self.MAX_PLAYERS):
            inputstructs[i].stickX = random.randint(-64, 64)
            inputstructs[i].stickY = random.randint(-64, 64)
            inputstructs[i].buttonInput = (ctypes.c_bool * 3)(*[random.choices([True, False],[39/40,1/40]) for _ in range(3)])
            
        self.pixelPointers = self.dll.step_pixels(inputstructs)
        for i in range(self.MAX_PLAYERS):
            pixelStruct = self.pixelPointers[i].contents
            self.imgs[i] = np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height * 3).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height, 3))

    def render(self):
        self.window.fill((0, 0, 0))
        for i in range(self.MAX_PLAYERS):
            pixelStruct = self.pixelPointers[i].contents
            self.imgs[i] = Image.fromarray(np.fromiter(pixelStruct.pixels, dtype=int, count=pixelStruct.width * pixelStruct.height * 3).astype(np.uint8).reshape((pixelStruct.width, pixelStruct.height, 3)))
            self.imgs[i] = self.imgs[i].transpose(Image.FLIP_TOP_BOTTOM)

            surface = pygame.image.fromstring(self.imgs[i].tobytes(), self.imgs[i].size, self.imgs[i].mode)
            self.window.blit(surface, ((i % self.N_SCREENS_WIDTH) * pixelStruct.height, (i // 5) * pixelStruct.width))

        pygame.display.flip()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

if __name__ == "__main__":
    env = SM64_ENV()
    # env.reset()
    done = False

    while not done:
        for i in range(100):
            actions = {}
            env.step(actions)
            env.render()
        print("RESET")
        env.reset()
    
