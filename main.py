import ctypes
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
from PIL import Image
import numpy as np
import copy
import random
funky = ctypes.CDLL(r"./build/us_pc/sm64.dll")


class gfxPixels(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ), 
        ("height",ctypes.c_int ), 
        ("width",ctypes.c_int ), 
    ]

funky.main_func()


funky.step_pixels.restype = ctypes.POINTER(ctypes.POINTER(gfxPixels))
# funky.step_ray_pixels.restype = gfxPixels


ray_mode = False

steps = 0
while True:
    if steps == 10:
        funky.makemariolol()
    steps += 1
    # funky.step()
    # print(steps)

    if ray_mode:
        pixelStruct = funky.step_ray_pixels()
    else:
        pixelStruct = funky.step_pixels()[random.randint(0,3)].contents

    if steps % 10 == 0:
        # pixelslist = [ pixelStruct.pixels[i] for i in range(pixelStruct.width * pixelStruct.height * 3) ]
        for i in range(4):
            if ray_mode:
                img = Image.fromarray(np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height).astype(np.uint8).reshape(( pixelStruct.height,pixelStruct.width)))
                
                img = img.transpose(Image.TRANSPOSE)
                img = img.transpose(Image.ROTATE_180)
            else:
                img = Image.fromarray(np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height * 3).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height, 3)))
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img = img.resize((256,144))

        
        img.save("test.png")



# print function 




