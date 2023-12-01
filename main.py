import ctypes
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
from PIL import Image
import numpy as np
import copy
import random
import pygame

MAX_PLAYERS = 4
# Initialize Pygame
pygame.init()

# Set the width and height of the window
N_SCREENS_WIDTH = 5

window_width = 256 * N_SCREENS_WIDTH
window_height = 144 * MAX_PLAYERS // N_SCREENS_WIDTH + 1

# Create the window
window = pygame.display.set_mode((window_width, window_height))

# Set the window title
pygame.display.set_caption("mario command panel")

funky = ctypes.CDLL(r"./build/us_pc/sm64.dll")


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

funky.main_func()

funky.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(gfxPixels))
funky.step_pixels.argtypes = [inputStruct * MAX_PLAYERS]

# funky.step_ray_pixels.restype = gfxPixels


imgs = list(range(MAX_PLAYERS))

steps = 0
running = True
while running:
    if steps == 10:
        funky.makemariolol()
    steps += 1

    start_time = time.time()
    

    inputstructs = (inputStruct * MAX_PLAYERS)()
    for i in range(MAX_PLAYERS):
        inputstructs[i].stickX = random.randint(-64, 64)
        inputstructs[i].stickY = random.randint(-64, 64)
        inputstructs[i].buttonInput = (ctypes.c_bool * 3)(*[random.choices([True, False],[39/40,1/40]) for _ in range(3)])
        
    # print(inputstructs[1].stickX, inputstructs[1].stickY, inputstructs[1].buttonInput[0], inputstructs[1].buttonInput[1], inputstructs[1].buttonInput[2])

    pixelPointers = funky.step_pixels(inputstructs)

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"{execution_time}")

    # if steps % 1  == 0:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    window.fill((0, 0, 0))


    for i in range(MAX_PLAYERS):
        pixelStruct = pixelPointers[i].contents

        imgs[i] = Image.fromarray(np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height * 3).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height, 3)))
        imgs[i] = imgs[i].transpose(Image.FLIP_TOP_BOTTOM)
        # img = img.resize((256,144))

        
        imgs[i].save(f"test{i}.png")
        surface = pygame.image.fromstring(imgs[i].tobytes(), imgs[i].size, imgs[i].mode)
        window.blit(surface, ((i % N_SCREENS_WIDTH) * pixelStruct.height, (i // 5) * pixelStruct.width))

    pygame.display.flip()




