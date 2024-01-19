from pettingzoo import ParallelEnv
import ctypes
import pygame
import random
from PIL import Image
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import functools
import math
import os
import platform
from collections import deque 
import time

class GAME_STATE_STRUCT(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ),
        ("pixelsHeight",ctypes.c_int ), 
        ("pixelsWidth",ctypes.c_int ),
        ("health",ctypes.c_int),
        ("posX",ctypes.c_float),
        ("posY",ctypes.c_float),
        ("posZ",ctypes.c_float),
        ("velX",ctypes.c_float),
        ("velY",ctypes.c_float),
        ("velZ",ctypes.c_float),
    ]

class INPUT_STRUCT(ctypes.Structure):
    _fields_ = [
        ("stickX",ctypes.c_short), 
        ("stickY",ctypes.c_short ), 
        ("buttonInput",ctypes.c_bool * 3), # goes A,B,Z
    ]

def make_action_struct(a):
    angleDegrees,A,B,Z = a
    inputStruct = INPUT_STRUCT()
    if angleDegrees == "noStick":
        inputStruct.stickX = 0
        inputStruct.stickY = 0
    else:
        inputStruct.stickX = round(64 * math.sin(angleDegrees * math.pi/180))
        inputStruct.stickY = round(64 * math.cos(angleDegrees * math.pi/180))
    inputStruct.buttonInput = (ctypes.c_bool * 3)(A,B,Z)
    return inputStruct


# window and dll outside of the class isn't ideal, but pettingzoo wrappers required the env to be pickleable for some reason


window = pygame.display.set_mode((100,100))

dirpath = os.path.dirname(__file__)

dll_name = "sm64.dll" if platform.platform() == "Windows" else "sm64"
dll = ctypes.CDLL( os.path.join(dirpath,"build","us_pc",dll_name) )     
class SM64_ENV(ParallelEnv):
    metadata = {
        "name": "sm64",
    }

    def __init__(self, FRAME_SKIP=4 , MAKE_OTHER_PLAYERS_INVISIBLE=True,PLAYER_COLLISION_TYPE=0, AUTO_RESET = False, ACTION_BOOK=[],
                 N_RENDER_COLUMNS=5, render_mode="forced", HIDE_AND_SEEK_MODE=False,
                 IMG_WIDTH=128, IMG_HEIGHT=72):
        self.render_mode = render_mode

        # if angleDegrees == "noStick" then there is no direction held
        # ie ["noStick",False,False,False] does nothing
        if ACTION_BOOK != []:
            self.action_book = ACTION_BOOK
        else:
            self.action_book = [
                # angleDegrees, A, B, Z
                # -----FORWARD
                [0,False,False,False],
                [0,True,False,False],
                # -----FORWARD RIGHT
                [30,False,False,False],
                # -----FORWARD LEFT
                [-30,False,False,False],
            ]

        # MAX_PLAYERS is decided when compiling
        dll.max_players_reminder.restype = ctypes.c_int
        self.MAX_PLAYERS = dll.max_players_reminder()

        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

        self.N_ACTIONS = len(self.action_book)

        self.FRAME_SKIP = FRAME_SKIP
        self.AUTO_RESET = AUTO_RESET

        self.N_RENDER_COLUMNS = N_RENDER_COLUMNS

        self.RENDER_WINDOW_WIDTH = self.IMG_WIDTH * self.N_RENDER_COLUMNS
        self.RENDER_WINDOW_HEIGHT = self.IMG_HEIGHT * ((self.MAX_PLAYERS // self.N_RENDER_COLUMNS) + 1)

        self.agents = [f"mario_{k}" for k in range(self.MAX_PLAYERS) ] 
        self.possible_agents = [f"mario_{k}" for k in range(self.MAX_PLAYERS)]

        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) } 
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }


        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]

        self.rewards = [0 for _ in range(self.MAX_PLAYERS)]

        pygame.init()
        window = pygame.display.set_mode((self.RENDER_WINDOW_WIDTH, self.RENDER_WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")


        
        dll.step_pixels.argtypes = [INPUT_STRUCT * self.MAX_PLAYERS , ctypes.c_int]
        dll.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(GAME_STATE_STRUCT))
        
        dll.main_func.argtypes = [ctypes.c_char_p,ctypes.c_char_p, ctypes.c_bool,ctypes.c_int,ctypes.c_bool,ctypes.c_int,ctypes.c_int]

        dll.main_func(dirpath.encode('utf-8'),dirpath.encode('utf-8'),MAKE_OTHER_PLAYERS_INVISIBLE,PLAYER_COLLISION_TYPE, HIDE_AND_SEEK_MODE,IMG_WIDTH,IMG_HEIGHT)
        actions = {agent: self.action_space(agent).sample() for agent in self.agents}
        for i in range(10):
            self.step(actions)
        # print("making marios")
        dll.makemariolol()
        self.reset()

    def reset(self, seed=None, options=None):
        dll.reset()

        self.agents = self.possible_agents.copy()
        # reset the image stacks
        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]
        
        for i in range(10):
            actions = {agent: self.action_space(agent).sample() for agent in self.agents}
            observations, rewards, terminations, truncations, infos = self.step(actions)

        return observations, infos

    def step(self, actions):
        inputStructs = (INPUT_STRUCT * self.MAX_PLAYERS)()
        for name in actions:
            inputStructs[self.AGENT_NAME_TO_INDEX[name]] = make_action_struct(self.action_book[actions[name]])

        gameStatePointers = dll.step_pixels(inputStructs,self.FRAME_SKIP)
        
            

        self.make_np_imgs(gameStatePointers)
        self.calc_rewards(gameStatePointers)

        observations = {a: self.np_imgs[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        rewards      = {a: self.rewards[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        infos        = {a: {}                                                                       for a in self.agents}
        terminations = {a: gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health == 0      for a in self.agents}
        truncations  = {a: False                                                                    for a in self.agents}
        # print([self.gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health for a in self.agents])


        if self.render_mode == "forced":
            self.render()
        # self.render()
        if self.AUTO_RESET and (any(terminations.values()) or all(truncations.values())):
            # self.agents = []
            observations, infos = self.reset()
            terminations = {a: False for a in self.agents}
            truncations  = {a: False for a in self.agents}
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="default"):
        imgs = [0 for i in range(self.MAX_PLAYERS)]
        for i in range(self.MAX_PLAYERS):
            # [0] gives newest image
            imgs[i] = Image.fromarray(self.np_imgs[i],'RGB')

        if mode == "tag":
            # put hiders and seekers together
            tmp = [0 for i in range(self.MAX_PLAYERS)]
            tmp[::2] = imgs[0:self.MAX_PLAYERS//2] 
            tmp[1::2] = imgs[self.MAX_PLAYERS//2:]
            imgs = tmp



        window.fill((0, 0, 0))
        for i in range(len(imgs)):
            # if you get a NULL pointer access, then self.MAX_PLAYERS doesn't line up with the C code
            tmp = imgs[i].convert("RGB")
            surface = pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode)
            window.blit(surface, ((i % self.N_RENDER_COLUMNS) * self.IMG_WIDTH, (i // self.N_RENDER_COLUMNS) * self.IMG_HEIGHT))

        pygame.display.flip()

    def calc_rewards(self,gameStatePointers):
        # placeholder reward function
        goalpos = (1770,1000,1986) #roughly around the chain chomp
        for i in range(self.MAX_PLAYERS):
            s = gameStatePointers[i].contents
            # if i == 0:
            #     print(s.posX,s.posZ,s.velX,s.velZ)
            self.rewards[i] = (30 - ( (s.posX - goalpos[0])**2  + (s.posZ - goalpos[2])**2 )/4000000 ) / 30


    def make_np_imgs(self,gameStatePointers):
        for i in range(self.MAX_PLAYERS):
            gameStateStruct = gameStatePointers[i].contents
            new_np_img = np.fromiter(gameStateStruct.pixels, dtype=int, count=gameStateStruct.pixelsWidth * gameStateStruct.pixelsHeight * 3).reshape((gameStateStruct.pixelsWidth, gameStateStruct.pixelsHeight, 3))
            new_np_img = np.flipud(new_np_img).astype(np.uint8)

            self.np_imgs[i] = new_np_img

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=255, shape=(self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8)
        
    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.N_ACTIONS)

if __name__ == "__main__":
    env = SM64_ENV(FRAME_SKIP=4)

    done = False
    while not done:
        for i in range(1000000):
            list_actions = [0 for _ in range(env.MAX_PLAYERS)]
            # if i % 10 == 0:
            #     list_actions = [2 for _ in range(env.MAX_PLAYERS)]
            # if i % 10 == 1:
            #     list_actions = [1 for _ in range(env.MAX_PLAYERS)]
                
            # list_actions = [random.randint(0,env.N_ACTIONS-1) for _ in range(env.MAX_PLAYERS)]
            actions = {f"mario_{k}": list_actions[k] for k in range(env.MAX_PLAYERS) }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print("SHAPE: ",observations["mario0"].shape)
            env.render()

        print("RESET")
        env.reset()
    
