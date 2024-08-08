# from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from env.sm64_env_mixed import SM64_ENV_MIXED
from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID

from tqdm import tqdm
import supersuit as ss
import numpy as np
import multiprocessing
import gymnasium
import time
import random
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from scipy.special import gammainc
from matplotlib import pyplot as plt
# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure

from planner import PLANNER


ACTION_BOOK = [
    # -----FORWARD RIGHT
    [30,False,False,False],
    [30,True,False,False],
    [10,False,False,False],
    [10,True,False,False],

    # -----FORWARD LEFT
    [-30,False,False,False],
    [-30,True,False,False],
    [-10,False,False,False],
    [-10,True,False,False],
]



########################## BLACK AND WHITE FILTER

GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
def preprocess_img(obs, space):
    # observation is (image, numerical)
    # grayscale
    new_obs = (np.expand_dims((obs[0] @ GRAYSCALE_WEIGHTS), axis=-1).astype(np.uint8), obs[1])
    # print(new_obs[0].shape, new_obs[1].shape)

    return new_obs

def preprocess_space(space):
    # observation is (image, numerical) 
    img_shape = space[0].shape
    new_img_shape = (img_shape[0],img_shape[1], 1)

    space = gymnasium.spaces.Tuple([gymnasium.spaces.Box(low=0, high=255, shape=new_img_shape, dtype=np.uint8), 
                                    space[1]])
    return space
##############################################




def hard_coded_agent(obs, goalpos=(0,100,0)):
    # print(obs)
    x,y,z,vx,vy,vz,abs_v = obs
    x,y,z = x*8000, y*8000, z*8000
    vx,vy,vz = vx*50, vy*50, vz*50
    goal_dir = (goalpos[0] - x, goalpos[1] - y, goalpos[2] - z)
    go_left = goal_dir[0]*vz - goal_dir[2]*vx
    if go_left > 0:
        action = 4 # 30 left
    else:
        action = 0 # 30 right
    if goal_dir[1] > 0 and random.random() < 0.1:
        action += 1 # jump 

    p.add_circle((x,y,z))
    # return action
    return np.random.randint(0, len(ACTION_BOOK) - 1)



if __name__ == "__main__":
    # envs = SM64_ENV_CURIOSITY(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK,
    #                             NODES_MAX=3000, NODE_RADIUS= 400, NODES_MAX_VISITS=400, NODE_MAX_HEIGHT_ABOVE_GROUND=1000,
    #                             MAKE_OTHER_PLAYERS_INVISIBLE=True, IMG_WIDTH=128, IMG_HEIGHT=72)
    # envs = ss.black_death_v3(envs)
    # envs = ss.clip_reward_v0(envs, lower_bound=-1, upper_bound=1)
    # envs = ss.color_reduction_v0(envs, mode="full")
    # envs = ss.frame_stack_v1(envs, 1)
    # envs = ss.pettingzoo_env_to_vec_env_v1(envs)
    p = PLANNER()

    env = SM64_ENV_MIXED(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK)

    envs = ss.observation_lambda_v0(env, preprocess_img, preprocess_space)
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)
    envs.black_death = True
    # num_dll = multiprocessing.cpu_count()
    num_dll = 3

    envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
    

    observations, _ = envs.reset()
    img_obs, numerical_obs = observations
    cols = 4
    renderer = SM64_ENV_RENDER_GRID(128, 72, N_RENDER_COLUMNS=cols,N_RENDER_ROWS=(envs.num_envs//cols) + 1, coloured=False, mode="normal")
    INIT_HP = {
        "MAX_EPISODES": 1,
        "MAX_EPISODE_LENGTH": 1000,
    }
    for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
        for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
            # print(f"-----------------------{envs.num_envs} {num_dll}")
            # actions = np.random.randint(0, len(ACTION_BOOK) - 1, size=envs.num_envs)
            actions = np.array([hard_coded_agent(obs) for obs in numerical_obs])
            # actions[0] = len(ACTION_BOOK) - 2
            observations, rewards, terminations, truncations, infos = envs.step(actions)
            img_obs, numerical_obs = observations
            renderer.render_game(observations[0])
            # time.sleep(0.016)
        envs.reset()
        # x,y,z,vx,vy,vz,abs_v = numerical_obs[0]
        # print([8000*x,8000*y,8000*z,50*vx,50*vy,50*vz,abs_v])

    print("Passed test :)")
    # Plotting the sampled circles
    p.plot_F(draw_edges=False)