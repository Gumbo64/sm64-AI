from pettingzoo import ParallelEnv
import ctypes
# import pygame
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
import shutil
import uuid

class GAME_STATE_STRUCT(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ),
        ("pixelsHeight",ctypes.c_int ), 
        ("pixelsWidth",ctypes.c_int ),
        ("health",ctypes.c_int),
        ("deathNotice",ctypes.c_int),
        ("posX",ctypes.c_float),
        ("posY",ctypes.c_float),
        ("posZ",ctypes.c_float),
        ("velX",ctypes.c_float),
        ("velY",ctypes.c_float),
        ("velZ",ctypes.c_float),
        ("heightAboveGround",ctypes.c_float),
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


# window = pygame.display.set_mode((100,100))

dirpath = os.path.dirname(__file__)
dll_collection = {}

class SM64_ENV(ParallelEnv):
    metadata = {
        "name": "sm64",
    }

    def __init__(self, FRAME_SKIP=4 , MAKE_OTHER_PLAYERS_INVISIBLE=True,PLAYER_COLLISION_TYPE=0, AUTO_RESET = False, ACTION_BOOK=[],
                 N_RENDER_COLUMNS=5, render_mode=None,HIDE_AND_SEEK_MODE=False, COMPASS_ENABLED=False, TOP_DOWN_CAMERA=False,
                 IMG_WIDTH=84, IMG_HEIGHT=84):
        self.render_mode = render_mode
        self.np_random = np.random.Generator(np.random.PCG64())
        # if angleDegrees == "noStick" then there is no direction held
        # ie ["noStick",False,False,False] does nothing
        if ACTION_BOOK != []:
            self.ACTION_BOOK = ACTION_BOOK
        else:
            self.ACTION_BOOK = [
                # angleDegrees, A, B, Z
                # -----FORWARD
                [0,False,False,False],
                [0,True,False,False],
                # -----FORWARD RIGHT
                [30,False,False,False],
                # -----FORWARD LEFT
                [-30,False,False,False],
            ]

        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.N_ACTIONS = len(self.ACTION_BOOK)
        self.FRAME_SKIP = FRAME_SKIP
        self.AUTO_RESET = AUTO_RESET
        self.N_RENDER_COLUMNS = N_RENDER_COLUMNS
        self.MAKE_OTHER_PLAYERS_INVISIBLE = MAKE_OTHER_PLAYERS_INVISIBLE
        self.PLAYER_COLLISION_TYPE = PLAYER_COLLISION_TYPE
        self.HIDE_AND_SEEK_MODE = HIDE_AND_SEEK_MODE
        self.COMPASS_ENABLED = COMPASS_ENABLED
        self.TOP_DOWN_CAMERA = TOP_DOWN_CAMERA
        
        # managing DLLs
        my_id = id(self)
        original_dll_name = "sm64.dll" if "Windows" in platform.platform() else "sm64"
        new_dll_name = f"sm64_{my_id}.dll" if "Windows" in platform.platform() else f"sm64_{my_id}"
        old_path = os.path.join(dirpath,"build","us_pc",original_dll_name)
        new_path = os.path.join(dirpath,"build","us_pc",new_dll_name)
        shutil.copyfile(old_path,new_path)

        self.dll = ctypes.CDLL(new_path)
        self.dll.max_players_reminder.restype = ctypes.c_int
        self.MAX_PLAYERS = self.dll.max_players_reminder()

        self.dll.set_compass_targets.argtypes = [ (3 * ctypes.c_float)* self.MAX_PLAYERS]
        self.dll.step_pixels.argtypes = [INPUT_STRUCT * self.MAX_PLAYERS , ctypes.c_int]
        self.dll.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(GAME_STATE_STRUCT))
        self.dll.main_func.argtypes = [ctypes.c_char_p,ctypes.c_char_p, ctypes.c_bool,ctypes.c_int,ctypes.c_bool,ctypes.c_bool,ctypes.c_int,ctypes.c_int, ctypes.c_bool]
        self.dll.main_func(dirpath.encode('utf-8'),dirpath.encode('utf-8'),self.MAKE_OTHER_PLAYERS_INVISIBLE,self.PLAYER_COLLISION_TYPE,self.HIDE_AND_SEEK_MODE, self.COMPASS_ENABLED,self.IMG_WIDTH,self.IMG_HEIGHT, self.TOP_DOWN_CAMERA)
        
        self.init_with_max_players()
        actions = {agent: self.action_space(agent).sample() for agent in self.agents}
        for i in range(10):
            self.step(actions)
        self.dll.makemariolol()
    
    def init_with_max_players(self):
        # initialise the rest of the variables that depend on MAX_PLAYERS
        self.RENDER_WINDOW_WIDTH = self.IMG_WIDTH * self.N_RENDER_COLUMNS
        self.RENDER_WINDOW_HEIGHT = self.IMG_HEIGHT * ((self.MAX_PLAYERS // self.N_RENDER_COLUMNS) + 1)
        self.agents = [f"mario_{k}" for k in range(self.MAX_PLAYERS) ] 
        self.possible_agents = [f"mario_{k}" for k in range(self.MAX_PLAYERS)]
        self.AGENT_NAME_TO_INDEX = {self.agents[k]: k for k in range(self.MAX_PLAYERS) } 
        self.INDEX_TO_AGENT_NAME = {k: self.agents[k] for k in range(self.MAX_PLAYERS) }
        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]
        self.rewards = [0 for _ in range(self.MAX_PLAYERS)]

        # pygame.init()
        # self.window = pygame.display.set_mode((self.RENDER_WINDOW_WIDTH, self.RENDER_WINDOW_HEIGHT))
        # pygame.display.set_caption("mario command panel")

    def __getstate__(self):
        all_state = self.__dict__.copy()
        new_state = {}
        # Don't pickle the window nor dll
        starting_fields = ["FRAME_SKIP","MAKE_OTHER_PLAYERS_INVISIBLE","PLAYER_COLLISION_TYPE","AUTO_RESET","ACTION_BOOK","N_RENDER_COLUMNS","render_mode","HIDE_AND_SEEK_MODE","COMPASS_ENABLED","TOP_DOWN_CAMERA","IMG_WIDTH","IMG_HEIGHT"]
        for field in starting_fields:
            new_state[field] = all_state[field]
        return new_state

    def __setstate__(self, state):
        # just remake the whole thing bro
        state_dict = {}
        state_dict.update(state)
        self.__init__(**state_dict)
    def reset(self, seed=None, options=None, random_moves=10):
        self.agents = self.possible_agents.copy()
        # reset the image stacks
        self.np_imgs = [np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8) for i in range(self.MAX_PLAYERS)]


        self.dll.reset()

        if random_moves == 0:
            actions = {agent: 0 for agent in self.agents}
            observations, rewards, terminations, truncations, infos = self.step(actions)
        else:
            for i in range(random_moves):
                actions = {agent: self.action_space(agent).sample() for agent in self.agents}
                observations, rewards, terminations, truncations, infos = self.step(actions)

        return observations, infos

    def step(self, actions):
        inputStructs = (INPUT_STRUCT * self.MAX_PLAYERS)()
        for name in actions:
            inputStructs[self.AGENT_NAME_TO_INDEX[name]] = make_action_struct(self.ACTION_BOOK[actions[name]])
        
        gameStatePointers = self.dll.step_pixels(inputStructs,self.FRAME_SKIP)
        
        self.make_np_imgs(gameStatePointers)
        self.calc_agent_rewards(gameStatePointers)
        self.make_infos(gameStatePointers)

        observations = {a: self.np_imgs[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        rewards      = {a: self.rewards[ self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        infos        = {a: self.infos[   self.AGENT_NAME_TO_INDEX[a] ]                              for a in self.agents}
        terminations = {a: gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health == 0      for a in self.agents}
        truncations  = {a: False                                                                    for a in self.agents}
        # print([self.gameStatePointers[self.AGENT_NAME_TO_INDEX[a]].contents.health for a in self.agents])


        # if self.render_mode == "forced":
        #     self.render()
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="default"):
        # imgs = [0 for i in range(self.MAX_PLAYERS)]
        # for i in range(self.MAX_PLAYERS):
        #     # [0] gives newest image
        #     imgs[i] = Image.fromarray(self.np_imgs[i],'RGB')

        # if mode == "tag":
        #     # put hiders and seekers together
        #     tmp = [0 for i in range(self.MAX_PLAYERS)]
        #     tmp[::2] = imgs[0:self.MAX_PLAYERS//2] 
        #     tmp[1::2] = imgs[self.MAX_PLAYERS//2:]
        #     imgs = tmp



        # self.window.fill((0, 0, 0))
        # for i in range(len(imgs)):
        #     # if you get a NULL pointer access, then self.MAX_PLAYERS doesn't line up with the C code
        #     tmp = imgs[i].convert("RGB")
        #     surface = pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode)
        #     self.window.blit(surface, ((i % self.N_RENDER_COLUMNS) * self.IMG_WIDTH, (i // self.N_RENDER_COLUMNS) * self.IMG_HEIGHT))

        # pygame.display.flip()
        
        pass

    def calc_agent_rewards(self,gameStatePointers):
        # placeholder reward function
        goalpos = (1770,1000,1986) #roughly around the chain chomp
        for i in range(self.MAX_PLAYERS):
            s = gameStatePointers[i].contents
            # if i == 0:
            #     print(s.posX,s.posZ,s.velX,s.velZ)
            self.rewards[i] = (30 - ( (s.posX - goalpos[0])**2  + (s.posZ - goalpos[2])**2 )/4000000 ) / 30

    def make_infos(self,gameStatePointers):
        pos = [(gameStatePointers[i].contents.posX,gameStatePointers[i].contents.posY,gameStatePointers[i].contents.posZ) for i in range(self.MAX_PLAYERS)]
        self.infos = []
        for i in range(self.MAX_PLAYERS):
            state = gameStatePointers[i].contents
            self.infos.append({"pos":pos[i], "died":state.deathNotice == 1, "health":state.health, "heightAboveGround":state.heightAboveGround, "vel":(state.velX,state.velY,state.velZ)})

    
    def set_compass_targets(self, targets):

        # Convert numpy array to ctypes array
        ctypes_targets = ((3 * ctypes.c_float)* self.MAX_PLAYERS)()
        for i in range(self.MAX_PLAYERS):
            ctypes_vec3f = (3 * ctypes.c_float)()
            for j in range(3):
                ctypes_vec3f[j] = targets[i][j]
            ctypes_targets[i] = ctypes_vec3f
        
        self.dll.set_compass_targets(ctypes_targets)

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

