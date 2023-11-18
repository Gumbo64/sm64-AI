import ctypes
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
from PIL import Image
import numpy as np
import copy

funky = ctypes.CDLL(r"./build/us_pc/sm64.dll")


class gfxPixels(ctypes.Structure):
    _fields_ = [
        ("pixels",ctypes.POINTER(ctypes.c_ubyte) ), 
        ("height",ctypes.c_int ), 
        ("width",ctypes.c_int ), 
    ]

funky.main_func()


funky.step_pixels.restype = gfxPixels




steps = 0
while True:
    if steps == 10:
        funky.makemariolol()
    steps += 1

    
    if steps % 1 == 0:
        pixelStruct = funky.step_pixels()
        # pixelslist = [ pixelStruct.pixels[i] for i in range(pixelStruct.width * pixelStruct.height * 3) ]
        # img = Image.fromarray(np.asarray(pixelslist).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height, 3)))

        # pixelslist = [ pixelStruct.pixels[i] for i in range(pixelStruct.width * pixelStruct.height * 3) ]
        # img = Image.fromarray(np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height * 3).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height, 3)))

        # img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # img = img.resize((256,144))


        img = Image.fromarray(np.fromiter(pixelStruct.pixels,dtype=int,count=pixelStruct.width * pixelStruct.height).astype(np.uint8).reshape(( pixelStruct.width,pixelStruct.height)))
        img = img.transpose(Image.TRANSPOSE)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save("test.png")
        # time.sleep(4)
    else:
        funky.step_headless()

# print function 




