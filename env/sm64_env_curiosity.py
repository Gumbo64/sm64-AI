from .sm64_env import SM64_ENV

import numpy as np
import random

from collections import deque 
# import multiprocessing
# from itertools import starmap

import gzip
import numpy as np
# import time
def C(x):
    b = x.tobytes()
    compressed = gzip.compress(b)
    return len(compressed)

def NCD(x,y):
    xy = np.concatenate([x,y])
    return ( C(xy) - min(C(x), C(y)) ) / max(C(x), C(y))

def NCD_2(x, y, C_x, C_y):   
    xy = np.concatenate([x, y])
    return (C(xy) - min(C_x, C_y)) / max(C_x, C_y)

        # with multiprocessing.Pool(processes=8) as pool:
                    # NCDs = starmap(NCD, [(self.np_imgs[i], sample_experiences[k]) for k in range(len(sample_experiences))])
        # result = np.average(np.fromiter(NCDs,dtype=int))


class SM64_ENV_CURIOSITY(SM64_ENV):
    def __init__(self, FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=True, PLAYER_COLLISION_TYPE=0, AUTO_RESET=False, N_RENDER_COLUMNS=2, render_mode="forced", HIDE_AND_SEEK_MODE=False, IMG_WIDTH=128, IMG_HEIGHT=72,
                 MAX_EXPERIENCE_BUFFER_SIZE=10000, N_SAMPLES=32 ):
        self.N_SAMPLES = N_SAMPLES
        # placeholder
        self.experience_buffer = deque([], maxlen=MAX_EXPERIENCE_BUFFER_SIZE)
        self.C_buffer = deque([], maxlen=MAX_EXPERIENCE_BUFFER_SIZE)
        super(SM64_ENV_CURIOSITY, self).__init__(FRAME_SKIP, MAKE_OTHER_PLAYERS_INVISIBLE, PLAYER_COLLISION_TYPE, AUTO_RESET, N_RENDER_COLUMNS, render_mode, HIDE_AND_SEEK_MODE, IMG_WIDTH, IMG_HEIGHT)
    
    def calc_rewards(self, gameStatePointers):
        # start_time = time.time()

        # np_imgs gives the current frame for each player, no frame stacking, green channel only
        
        current_imgs = [img[:,:,1] for img in self.np_imgs]
        current_Cs = [C(obs) for obs in current_imgs]

        self.experience_buffer += current_imgs
        self.C_buffer += current_Cs


        n_samples = min(self.N_SAMPLES, len(self.experience_buffer))
        sample_indices = np.random.randint(0, len(self.experience_buffer), n_samples)
        # exp_sample_concat = np.concatenate(sample_experiences,axis=0)

        for i in range(self.MAX_PLAYERS): 
            NCDs = [NCD_2(current_imgs[i], self.experience_buffer[sample_indices[k]], current_Cs[i] , self.C_buffer[sample_indices[k]]) for k in range(n_samples)]
            self.rewards[i] = (np.average(NCDs) - 0.9) * 5



        # end_time = time.time()
        # print(self.rewards)
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")
