from pettingzoo import ParallelEnv
import ctypes
import pygame
import random
from PIL import Image
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import functools
import math

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

def make_action(angleDegrees,A,B,Z):
    inputstruct = inputStruct()
    if angleDegrees == "noStick":
        inputstruct.stickX = 0
        inputstruct.stickY = 0
    else:
        inputstruct.stickX = round(64 * math.sin(angleDegrees * math.pi/180))
        inputstruct.stickY = round(64 * math.cos(angleDegrees * math.pi/180))
    inputstruct.buttonInput = (ctypes.c_bool * 3)(A,B,Z)
    return inputstruct

class SM64_ENV(ParallelEnv):
    metadata = {
        "name": "sm64",
    }

    def __init__(self):
        # angleDegrees, A, B, Z
        # if angleDegrees == "noStick" then there is no direction held
        self.action_book = [
            # -----FORWARD
            # None
            make_action(0,False,False,False),
            # Jump
            make_action(0,True,False,False),
            # Longjump
            make_action(0,True,False,True),
            # Dive
            make_action(0,False,True,False),

            # -----FORWARD RIGHT
            # None
            make_action(30,False,False,False),
            # Jump
            make_action(30,True,False,False),

            # -----FORWARD LEFT
            # None
            make_action(-30,False,False,False),
            # Jump
            make_action(-30,True,False,False),

            # -----BACKWARDS
            # None
            make_action(180,False,False,False),
            # Jump
            make_action(180,True,False,False),

            # ----- NO STICK (no direction held)
            # None
            make_action("noStick",False,False,False),
            # Groundpound
            make_action("noStick",False,False,True),
        ]
        self.N_ACTIONS = len(self.action_book)

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
            actions = [random.randint(0,self.N_ACTIONS-1) for _ in range(self.MAX_PLAYERS)]
            self.step(actions)
        print("making marios")
        self.dll.makemariolol()
        self.reset()

    def reset(self, seed=None, options=None):
        self.dll.reset()

    def step(self, actions):
        inputstructs = (inputStruct * self.MAX_PLAYERS)()
        for i in range(self.MAX_PLAYERS):
            inputstructs[i] = self.action_book[actions[i]]

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
        for i in range(1000000):
            actions = [0 for _ in range(env.MAX_PLAYERS)]
            if i % 10 == 0:
                actions = [2 for _ in range(env.MAX_PLAYERS)]
                

            
            # actions = [random.randint(0,env.N_ACTIONS-1) for _ in range(env.MAX_PLAYERS)]
            env.step(actions)
            env.render()
        print("RESET")
        env.reset()
    
