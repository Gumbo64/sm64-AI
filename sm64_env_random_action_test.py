from env.sm64_env import SM64_ENV
from env.sm64_env_tag import SM64_ENV_TAG
from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from env.sm64_env_rrt import SM64_ENV_RRT
from tqdm import tqdm
import supersuit as ss
import numpy as np
# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure


ACTION_BOOK = [
    # angle, A, B, Z
    # angle 0 is vertical
    # 8 directions
    [90,False,False,False],
    [-90,False,False,False],
    [0,False,False,False],
    [180,False,False,False],
    [45,False,False,False],
    [-45,False,False,False],
    [135,False,False,False],
    [-135,False,False,False],
    # 8 directions + jump
    [90,True,False,False],
    [-90,True,False,False],
    [0,True,False,False],
    [180,True,False,False],
    [45,True,False,False],
    [-45,True,False,False],
    [135,True,False,False],
    [-135,True,False,False],
]

# env = SM64_ENV(FRAME_SKIP=4,N_RENDER_COLUMNS=4, MAKE_OTHER_PLAYERS_INVISIBLE=False, ACTION_BOOK=ACTION_BOOK)
# env = SM64_ENV_TAG(FRAME_SKIP=4,N_RENDER_COLUMNS=4, MAKE_OTHER_PLAYERS_INVISIBLE=True, ACTION_BOOK=ACTION_BOOK)
env = SM64_ENV_CURIOSITY(
    FRAME_SKIP=4,
    COMPASS_ENABLED=False,
    N_RENDER_COLUMNS=4, 
    MAKE_OTHER_PLAYERS_INVISIBLE=False, 
    ACTION_BOOK=ACTION_BOOK,
    IMG_WIDTH=84,
    IMG_HEIGHT=84,
    TOP_DOWN_CAMERA=True
)

# env = SM64_ENV_RRT(render_mode="forced",FRAME_SKIP=4,COMPASS_ENABLED=True, N_RENDER_COLUMNS=4, MAKE_OTHER_PLAYERS_INVISIBLE=False, ACTION_BOOK=ACTION_BOOK)
# env.set_compass_targets(np.array([[-5272.5458984375, 0.0, 6842.7919921875] for _ in range(env.MAX_PLAYERS)]))

env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="full")
# env = ss.resize_v1(env, x_size=128, y_size=72)
env = ss.frame_stack_v1(env, 4,stack_dim=0)
env.reset()


env2 = SM64_ENV_CURIOSITY(
    FRAME_SKIP=4,
    COMPASS_ENABLED=False,
    N_RENDER_COLUMNS=4, 
    MAKE_OTHER_PLAYERS_INVISIBLE=False, 
    ACTION_BOOK=ACTION_BOOK,
    IMG_WIDTH=84,
    IMG_HEIGHT=84,
    TOP_DOWN_CAMERA=True
)
env2 = ss.clip_reward_v0(env2, lower_bound=-1, upper_bound=1)
env2 = ss.color_reduction_v0(env2, mode="full")
env2 = ss.frame_stack_v1(env2, 4,stack_dim=0)
env2.reset()


INIT_HP = {
    "MAX_EPISODES": 1000000,
    "MAX_EPISODE_LENGTH": 200,
}
for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
    for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        observations2, rewards2, terminations2, truncations2, infos2 = env2.step(actions)
    env.reset()
    env2.reset()
    

print("Passed test :)")