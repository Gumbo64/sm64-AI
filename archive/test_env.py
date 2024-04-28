from pettingzoo.test import parallel_api_test
from env.sm64_env import SM64_ENV
env = SM64_ENV(GRAYSCALE=False,N_ACTION_REPEAT=4)
parallel_api_test(env, num_cycles=1000)