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

    def __init__(self, FRAME_SKIP=1 , MAKE_OTHER_PLAYERS_INVISIBLE=True,PLAYER_COLLISION_TYPE=0, N_RENDER_COLUMNS=5, render_mode="normal"):
        self.render_mode = render_mode
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
        # this also needs to be changed in the c part (env/include/types.h) (and then compiled) to work. Maximum is 255 because of data types in c
        self.MAX_PLAYERS = 20
        self.num_envs = self.MAX_PLAYERS
        self.IMG_WIDTH = 128
        self.IMG_HEIGHT = 72

        self.N_ACTIONS = len(self.action_book)

        self.FRAME_SKIP = FRAME_SKIP


        self.N_RENDER_COLUMNS = N_RENDER_COLUMNS

        self.RENDER_WINDOW_WIDTH = self.IMG_WIDTH * self.N_RENDER_COLUMNS
        self.RENDER_WINDOW_HEIGHT = self.IMG_HEIGHT * self.MAX_PLAYERS // self.N_RENDER_COLUMNS + 1

        self.agents = [f"mario{k}" for k in range(self.MAX_PLAYERS) ]
        self.possible_agents = [f"mario{k}" for k in range(self.MAX_PLAYERS)]

        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) }
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }


        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]

        self.rewards = [0 for _ in range(self.MAX_PLAYERS)]

        pygame.init()
        self.window = pygame.display.set_mode((self.RENDER_WINDOW_WIDTH, self.RENDER_WINDOW_HEIGHT))
        pygame.display.set_caption("mario command panel")

        dirpath = os.path.dirname(__file__)

        dll_name = "sm64.dll" if platform.platform() == "Windows" else "sm64"
        self.dll = ctypes.CDLL( os.path.join(dirpath,"build","us_pc",dll_name) )     
        
        self.dll.step_pixels.argtypes = [INPUT_STRUCT * self.MAX_PLAYERS , ctypes.c_int]
        self.dll.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(GAME_STATE_STRUCT))
        
        self.dll.main_func.argtypes = [ctypes.c_char_p,ctypes.c_char_p, ctypes.c_bool,ctypes.c_int]

        self.dll.main_func(dirpath.encode('utf-8'),dirpath.encode('utf-8'),MAKE_OTHER_PLAYERS_INVISIBLE,PLAYER_COLLISION_TYPE)
        actions = {agent: self.action_space(agent).sample() for agent in self.agents}
        for i in range(10):
            self.step(actions)
        # print("making marios")
        self.dll.makemariolol()
        self.reset()

    def reset(self, seed=None, options=None):
        self.dll.reset()

        self.agents = self.possible_agents.copy()
        # reset the image stacks
        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]
        

        actions = {agent: self.action_space(agent).sample() for agent in self.agents}
        observations, rewards, terminations, truncations, infos = self.step(actions)

        return observations, infos

    def step(self, actions):
        inputStructs = (INPUT_STRUCT * self.MAX_PLAYERS)()
        for name in actions:
            inputStructs[self.AGENT_NAME_TO_INDEX[name]] = self.action_book[actions[name]]

        self.gameStatePointers = self.dll.step_pixels(inputStructs,self.FRAME_SKIP)

        self.make_np_imgs()
        self.calc_rewards()

        observations = {a: self.np_imgs[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        rewards      = {a: self.rewards[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        infos        = {a: {}                                                                       for a in self.agents}
        terminations = {a: self.gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health == 0 for a in self.agents}
        truncations  = {a: False                                                                    for a in self.agents}
        # print([self.gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health for a in self.agents])
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        if self.render_mode == "forced":
            self.render()
        # self.render()
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        imgs = [0 for i in range(self.MAX_PLAYERS)]
        for i in range(self.MAX_PLAYERS):
            # [0] gives newest image
            imgs[i] = Image.fromarray(self.np_imgs[i],'RGB')


        self.window.fill((0, 0, 0))
        for i in range(len(imgs)):
            gameStateStruct = self.gameStatePointers[i].contents
            tmp = imgs[i].convert("RGB")
            surface = pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode)
            self.window.blit(surface, ((i % self.N_RENDER_COLUMNS) * gameStateStruct.pixelsHeight, (i // self.N_RENDER_COLUMNS) * gameStateStruct.pixelsWidth))

        pygame.display.flip()

    def calc_rewards(self):
        # placeholder reward function
        goalpos = (1770,1000,1986) #roughly around the chain chomp
        for i in range(self.MAX_PLAYERS):
            s = self.gameStatePointers[i].contents
            # if i == 0:
            #     print(s.posX,s.posZ)
            self.rewards[i] = (30 - ( (s.posX - goalpos[0])**2  + (s.posZ - goalpos[2])**2 )/4000000 ) / 30
    

    def make_np_imgs(self):
        for i in range(self.MAX_PLAYERS):
            gameStateStruct = self.gameStatePointers[i].contents
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
            if i % 10 == 0:
                list_actions = [2 for _ in range(env.MAX_PLAYERS)]
            if i % 10 == 1:
                list_actions = [1 for _ in range(env.MAX_PLAYERS)]
                
            list_actions = [random.randint(0,env.N_ACTIONS-1) for _ in range(env.MAX_PLAYERS)]
            actions = {f"mario{k}": list_actions[k] for k in range(env.MAX_PLAYERS) }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print("SHAPE: ",observations["mario0"].shape)
            env.render()

        print("RESET")
        env.reset()
    
