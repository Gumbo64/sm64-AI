from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from tqdm import tqdm
import supersuit as ss


from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID
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


env = SM64_ENV_CURIOSITY(
    FRAME_SKIP=4,
    COMPASS_ENABLED=False,
    MAKE_OTHER_PLAYERS_INVISIBLE=False, 
    ACTION_BOOK=ACTION_BOOK,
    IMG_WIDTH=84,
    IMG_HEIGHT=84,
    TOP_DOWN_CAMERA=False
)


env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="full")
# env = ss.resize_v1(env, x_size=128, y_size=72)
env = ss.frame_stack_v1(env, 4)
env.reset()

INIT_HP = {
    "MAX_EPISODES": 1000000,
    "MAX_EPISODE_LENGTH": 200,
}

renderer = SM64_ENV_RENDER_GRID(84, 84, mode="normal")

for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
    for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        renderer.render_game(observations)
    env.reset()
    

    

print("Passed test :)")