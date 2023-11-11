from ctypes import *
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
funky = CDLL(r"./build/us_pc/sm64.dll")



funky.main_func()



for i in range(10):
    a = funky.step()

funky.makemariolol()

while True:
    a = funky.step()

    # print(f"{list(a.faceAngle)}")

