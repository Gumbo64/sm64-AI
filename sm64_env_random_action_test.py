from env.sm64_env import SM64_ENV
from tqdm import trange

env = SM64_ENV(GRAYSCALE=True,N_ACTION_REPEAT=4, N_STACKED_FRAMES=4)
env.reset()
INIT_HP = {
    "MAX_EPISODES": 10,
    "MAX_EPISODE_LENGTH": 100,
}
for idx_epi in trange(INIT_HP["MAX_EPISODES"]):
    for i in range(INIT_HP["MAX_EPISODE_LENGTH"]):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        if not env.agents:
            break
    env.reset()

print("Passed test :)")