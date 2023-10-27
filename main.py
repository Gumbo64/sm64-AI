from ctypes import *
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
funky = CDLL(r"./build/us_pc/sm64.dll")


funky.main_func()

# funky.step.restype = MarioState


while True:

    # print("------------------")
    # inputmariolol = MarioInput(10,10,0)
    # funky.mario_input(inputmariolol)
    a = funky.step()
    # print(f"{list(a.faceAngle)}")

