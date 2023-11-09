from ctypes import *
# from MarioStateSimple import MarioState
# from MarioInputSimple import MarioInput
import time
funky = CDLL(r"./build/us_pc/sm64.dll")



funky.main_func()

# funky2.main_func()
# time.sleep(2)
# funky.step.restype = MarioState

# for i in range(100):

#     # print("------------------")
#     # inputmariolol = MarioInput(10,10,0)
#     # funky.mario_input(inputmariolol)
#     a = funky.step()

#     # print(f"{list(a.faceAngle)}")




# funky.makemariolol()

for i in range(100):

    # print("------------------")
    # inputmariolol = MarioInput(10,10,0)
    # funky.mario_input(inputmariolol)
    a = funky.step()

    # print(f"{list(a.faceAngle)}")

funky.makemariolol()

while True:
    a = funky.step()

    # print(f"{list(a.faceAngle)}")

