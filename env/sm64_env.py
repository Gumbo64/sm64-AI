from pettingzoo import ParallelEnv
import ctypes
import pygame
import random
from PIL import Image
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import functools
import math
import os
import platform

class GAME_STATE_STRUCT(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ), 
        ("pixelsHeight",ctypes.c_int ), 
        ("pixelsWidth",ctypes.c_int ),
        ("terminal",ctypes.c_bool),
        ("posX",ctypes.c_float),
        ("posY",ctypes.c_float),
        ("posZ",ctypes.c_float),
    ]

class INPUT_STRUCT(ctypes.Structure):
    _fields_ = [
        ("stickX",ctypes.c_short), 
        ("stickY",ctypes.c_short ), 
        ("buttonInput",ctypes.c_bool * 3), # goes A,B,Z
    ]

def make_action(angleDegrees,A,B,Z):
    inputStruct = INPUT_STRUCT()
    if angleDegrees == "noStick":
        inputStruct.stickX = 0
        inputStruct.stickY = 0
    else:
        inputStruct.stickX = round(64 * math.sin(angleDegrees * math.pi/180))
        inputStruct.stickY = round(64 * math.cos(angleDegrees * math.pi/180))
    inputStruct.buttonInput = (ctypes.c_bool * 3)(A,B,Z)
    return inputStruct

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
            # start longjump (crouch)
            make_action(0,False,False,True),
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

            # # ----- NO STICK (no direction held)
            # # None
            # make_action("noStick",False,False,False),
            # # Groundpound
            # make_action("noStick",False,False,True),
        ]
        self.N_ACTIONS = len(self.action_book)

        self.MAX_PLAYERS = 4

        self.agents = [f"mario{k}" for k in range(self.MAX_PLAYERS) ]
        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) }
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }

        self.N_SCREENS_WIDTH = 5
        self.WINDOW_WIDTH = 256 * self.N_SCREENS_WIDTH
        self.WINDOW_HEIGHT = 144 * self.MAX_PLAYERS // self.N_SCREENS_WIDTH + 1
        
        self.imgs = [i for i in range(self.MAX_PLAYERS)]
        self.np_imgs = [i for i in range(self.MAX_PLAYERS)]

        self.rewards = [0 for _ in range(self.MAX_PLAYERS)]

        pygame.init()
        self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")

        dirpath = os.path.dirname(__file__)

        dll_name = "sm64.dll" if platform.platform() == "Windows" else "sm64"
        self.dll = ctypes.CDLL( os.path.join(dirpath,"build","us_pc",dll_name) )     
        
        self.dll.step_pixels.argtypes = [INPUT_STRUCT * self.MAX_PLAYERS]
        self.dll.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(GAME_STATE_STRUCT))
        
        self.dll.main_func.argtypes = [ctypes.c_char_p,ctypes.c_char_p]

        self.dll.main_func(dirpath.encode('utf-8'),dirpath.encode('utf-8'))
        for i in range(10):
            self.step(self.sample_actions())
        print("making marios")
        self.dll.makemariolol()
        self.reset()

    def reset(self, seed=None, options=None):
        self.dll.reset()
        self.step(self.sample_actions())

    def step(self, actions):
        inputStructs = (INPUT_STRUCT * self.MAX_PLAYERS)()
        for name in actions:
            inputStructs[self.AGENT_NAME_TO_INDEX[name]] = self.action_book[actions[name]]
        self.gameStatePointers = self.dll.step_pixels(inputStructs)
        self.make_np_imgs()
        self.calc_rewards()

        observations = {a: self.np_imgs[ self.AGENT_NAME_TO_INDEX[a] ] for a in self.agents}
        rewards      = {a: self.rewards[ self.AGENT_NAME_TO_INDEX[a] ] for a in self.agents}
        infos        = {a: {}                                          for a in self.agents}
        terminations = {a: False                                       for a in self.agents}
        truncations  = {a: False                                       for a in self.agents}

        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        self.window.fill((0, 0, 0))
        self.make_imgs()
        for i in range(self.MAX_PLAYERS):
            gameStateStruct = self.gameStatePointers[i].contents
            surface = pygame.image.fromstring(self.imgs[i].tobytes(), self.imgs[i].size, self.imgs[i].mode)
            self.window.blit(surface, ((i % self.N_SCREENS_WIDTH) * gameStateStruct.pixelsHeight, (i // 5) * gameStateStruct.pixelsWidth))

        pygame.display.flip()

    def calc_rewards(self):
        goalpos = (-1770,500,1986)
        for i in range(self.MAX_PLAYERS):
            s = self.gameStatePointers[i].contents
            self.rewards[i] = math.sqrt( (s.posX - goalpos[0])**2  + (s.posZ - goalpos[2])**2 )
    

    def make_np_imgs(self):
        for i in range(self.MAX_PLAYERS):
            gameStateStruct = self.gameStatePointers[i].contents
            self.np_imgs[i] = np.fromiter(gameStateStruct.pixels, dtype=int, count=gameStateStruct.pixelsWidth * gameStateStruct.pixelsHeight * 3).astype(np.uint8).reshape((gameStateStruct.pixelsWidth, gameStateStruct.pixelsHeight, 3))

            # make the image grayscale (https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale)
            self.np_imgs[i] = np.dot(self.np_imgs[i][...,:3], [0.299, 0.587, 0.114])
            
            self.np_imgs[i] = np.flipud(self.np_imgs[i])

    def make_imgs(self):
        for i in range(self.MAX_PLAYERS):
            self.imgs[i] = Image.fromarray(self.np_imgs[i], "L")


    def sample_actions(self):
        list_actions = [0 for _ in range(self.MAX_PLAYERS)]
        # list_actions = [random.randint(0,self.N_ACTIONS-1) for _ in range(self.MAX_PLAYERS)]
        actions = {f"mario{k}": list_actions[k] for k in range(self.MAX_PLAYERS) }
        return actions

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
            list_actions = [0 for _ in range(env.MAX_PLAYERS)]
            if i % 10 == 0:
                list_actions = [2 for _ in range(env.MAX_PLAYERS)]
            if i % 10 == 1:
                list_actions = [1 for _ in range(env.MAX_PLAYERS)]
                

            
            # actions = [random.randint(0,env.N_ACTIONS-1) for _ in range(env.MAX_PLAYERS)]
            actions = {f"mario{k}": list_actions[k] for k in range(env.MAX_PLAYERS) }
            env.step(actions)
            env.render()
        print("RESET")
        env.reset()
    
