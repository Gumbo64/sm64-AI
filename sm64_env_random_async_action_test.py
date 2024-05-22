# from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from env.sm64_env_tag import SM64_ENV_TAG
from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID

from tqdm import tqdm
import supersuit as ss
import numpy as np
import multiprocessing
import gymnasium
import time
# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure


ACTION_BOOK = [
    # -----FORWARD
    # None
    [0,False,False,False],
    # Jump
    [0,True,False,False],
    # start longjump (crouch)
    [0,False,False,True],
    # Dive
    [0,False,True,False],

    # -----FORWARD RIGHT
    # None
    [30,False,False,False],
    [10,False,False,False],
    # Jump
    # [30,True,False,False],

    # -----FORWARD LEFT
    # None
    [-30,False,False,False],
    [-10,False,False,False],
    # Jump
    # [-30,True,False,False],

    # -----BACKWARDS
    # None
    [180,False,False,False],
    # Jump
    [180,True,False,False],

    # # ----- NO STICK (no direction held)
    # # None
    # ["noStick",False,False,False],
    # # Groundpound
    # ["noStick",False,False,True],
]
GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
def preprocess_img(obs, space):
    # observation is (image, numerical)
    # grayscale
    new_obs = (np.expand_dims((obs[0] @ GRAYSCALE_WEIGHTS),axis=-1), obs[1])
    # print(new_obs[0].shape, new_obs[1].shape)
    return new_obs

def preprocess_space(space):
    # observation is (image, numerical)
    img_shape = space.shape
    new_img_shape = (img_shape[0], img_shape[1], 1)
    space = gymnasium.spaces.Tuple([gymnasium.spaces.Box(low=0, high=255, shape=new_img_shape, dtype=np.float32), gymnasium.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)])
    return space
if __name__ == "__main__":
    # envs = SM64_ENV_CURIOSITY(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK,
    #                             NODES_MAX=3000, NODE_RADIUS= 400, NODES_MAX_VISITS=400, NODE_MAX_HEIGHT_ABOVE_GROUND=1000,
    #                             MAKE_OTHER_PLAYERS_INVISIBLE=True, IMG_WIDTH=128, IMG_HEIGHT=72)
    # envs = ss.black_death_v3(envs)
    # envs = ss.clip_reward_v0(envs, lower_bound=-1, upper_bound=1)
    # envs = ss.color_reduction_v0(envs, mode="full")
    # envs = ss.frame_stack_v1(envs, 1)
    # envs = ss.pettingzoo_env_to_vec_env_v1(envs)

    env = SM64_ENV_TAG(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK)

    envs = ss.observation_lambda_v0(env, preprocess_img, preprocess_space)
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)
    envs.black_death = True
    # num_dll = multiprocessing.cpu_count()
    num_dll = 1

    envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
    

    envs.reset()
    cols = 4
    renderer = SM64_ENV_RENDER_GRID(128, 72, N_RENDER_COLUMNS=cols,N_RENDER_ROWS=(envs.num_envs//cols) + 1, coloured=False, mode="normal")
    INIT_HP = {
        "MAX_EPISODES": 1000000,
        "MAX_EPISODE_LENGTH": 1000,
    }
    for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
        for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
            # print(f"-----------------------{envs.num_envs} {num_dll}")
            actions = np.random.randint(0, len(ACTION_BOOK) - 1, size=envs.num_envs)
            actions[1] = len(ACTION_BOOK) - 2
            observations, rewards, terminations, truncations, infos = envs.step(actions)
            renderer.render_game(observations[0])
            # time.sleep(0.1)
        envs.reset()

    print("Passed test :)")