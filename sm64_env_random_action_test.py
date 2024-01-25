from env.sm64_env import SM64_ENV
from env.sm64_env_tag import SM64_ENV_TAG
from tqdm import trange
import supersuit as ss

# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure

# Can test either of these
# env = SM64_ENV(FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=False)

ACTION_BOOK = [
    # angle, A, B, Z
    [10,False,False,False],
    # [30,False,False,False],
    [-10,False,False,False],
    # [-30,False,False,False],
]

env = SM64_ENV_TAG(FRAME_SKIP=4,N_RENDER_COLUMNS=4, MAKE_OTHER_PLAYERS_INVISIBLE=False, ACTION_BOOK=ACTION_BOOK)

env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="full")
# env = ss.resize_v1(env, x_size=128, y_size=72)
env = ss.frame_stack_v1(env, 4,stack_dim=0)
env.reset()
INIT_HP = {
    "MAX_EPISODES": 1000000,
    "MAX_EPISODE_LENGTH": 200,
}
for idx_epi in trange(INIT_HP["MAX_EPISODES"]):
    for i in range(INIT_HP["MAX_EPISODE_LENGTH"]):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(rewards)
        # print(observations["mario0"].shape)
        # env.render()
        if not env.agents:
            break
    env.reset()
    

print("Passed test :)")