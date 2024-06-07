from env.sm64_env_curiosity_mixedinput import SM64_ENV_CURIOSITY_MIXED
from tqdm import tqdm
import supersuit as ss


from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID
# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure
import gymnasium
import numpy as np


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


env = SM64_ENV_CURIOSITY_MIXED(
    FRAME_SKIP=4,
    COMPASS_ENABLED=False,
    MAKE_OTHER_PLAYERS_INVISIBLE=False, 
    ACTION_BOOK=ACTION_BOOK,
    IMG_WIDTH=128,
    IMG_HEIGHT=72,
    TOP_DOWN_CAMERA=False
)


envs = ss.observation_lambda_v0(env, preprocess_img, preprocess_space)
envs = ss.pettingzoo_env_to_vec_env_v1(envs)
envs.black_death = True

# num_dll = multiprocessing.cpu_count()
num_dll = 1

envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
envs.black_death = True

envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True


envs.reset()

INIT_HP = {
    "MAX_EPISODES": 1000000,
    "MAX_EPISODE_LENGTH": 200,
}

renderer = SM64_ENV_RENDER_GRID(128, 72, mode="normal")

for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
    for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
        # actions = {agent: envs.action_space(agent).sample() for agent in env.agents}
        actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
        observations, rewards, terminations, truncations, infos = envs.step(actions)
        renderer.render_game(observations[0])
    envs.reset()
    

    

print("Passed test :)")