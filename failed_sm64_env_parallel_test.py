from typing import Callable, List, Optional
from gym.core import Env
from env.sm64_env import SM64_ENV
from env.sm64_env_tag import SM64_ENV_TAG
from tqdm import trange
import supersuit as ss
import gymnasium as gym
import random
#import subprocvecenv
from stable_baselines3.common.vec_env import SubprocVecEnv
# use the SM64_ENV version of FRAME_SKIP, it will run fully in C without rendering, much faster.
# ngl might be very minor physics issues though, not 100% sure

# Can test either of these
# env = SM64_ENV(FRAME_SKIP=4, MAKE_OTHER_PLAYERS_INVISIBLE=False)


class SuperSubprocVecEnv(SubprocVecEnv):
    def __init__(self,*args, **kwargs):
        super(SubprocVecEnv, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(self.observation_space.low,self.observation_space.high,shape=[20] * self.observation_space.shape)
        self.action_space = gym.spaces.MultiDiscrete([20] * self.action_space.nvec.shape[0])
        pass

def make_env():
    env = SM64_ENV_TAG(FRAME_SKIP=4, N_RENDER_COLUMNS=4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="full")

    env = ss.frame_stack_v1(env, 4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return env
if __name__ == "__main__":
    # env = make_env()
    # Only works with 1 env at the same time unfortunately. This is because of CDLL, u can't open multiple instances of the same dll
    # Although it does work when they are in different cores? or processes? idk ray rllib did it somehow
    n = 5
    env = SuperSubprocVecEnv(env_fns=[make_env for _ in range(n)])
    # env = ss.concat_vec_envs_v1(make_env(), 2, num_cpus=99999, base_class="gymnasium")
    env.reset()
    INIT_HP = {
        "MAX_EPISODES": 1000000,
        "MAX_EPISODE_LENGTH": 200,
        "NUM_PLAYERS":20,
    }
    for idx_epi in trange(INIT_HP["MAX_EPISODES"]):
        for i in range(INIT_HP["MAX_EPISODE_LENGTH"]):


            actions = [random.randint(0,3) for _ in range(INIT_HP["NUM_PLAYERS"])]
            # actions = {"mario0":4,"mario1":9}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            print(rewards)
            # print(observations["mario0"].shape)
            # env.render()
            # if not env.agents:
            #     break
        env.reset()
        

    print("Passed test :)")