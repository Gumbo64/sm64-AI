from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID

from tqdm import tqdm
import supersuit as ss
import numpy as np
import multiprocessing


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

if __name__ == "__main__":
    envs = SM64_ENV_CURIOSITY(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK,
                                NODES_MAX=3000, NODE_RADIUS= 400, NODES_MAX_VISITS=400, NODE_MAX_HEIGHT_ABOVE_GROUND=1000,
                                MAKE_OTHER_PLAYERS_INVISIBLE=True, IMG_WIDTH=128, IMG_HEIGHT=72)
    envs = ss.black_death_v3(envs)
    envs = ss.clip_reward_v0(envs, lower_bound=-1, upper_bound=1)
    envs = ss.color_reduction_v0(envs, mode="full")
    envs = ss.frame_stack_v1(envs, 1)
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)

    # num_dll = multiprocessing.cpu_count()
    num_dll = 1

    envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
    envs.reset()
    renderer = SM64_ENV_RENDER_GRID(128, 72, N_RENDER_COLUMNS=5, mode="normal")
    INIT_HP = {
        "MAX_EPISODES": 1000000,
        "MAX_EPISODE_LENGTH": 3000,
    }
    for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
        for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
            # print(f"-----------------------{envs.num_envs} {num_dll}")
            actions = np.random.randint(0, len(ACTION_BOOK), size=envs.num_envs)
            observations, rewards, terminations, truncations, infos = envs.step(actions)
            renderer.render_game(observations)
        envs.reset()

    print("Passed test :)")