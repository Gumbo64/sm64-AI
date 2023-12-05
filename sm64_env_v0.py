from env.sm64_env import SM64_ENV
import random

env = SM64_ENV(GRAYSCALE=True,N_ACTION_REPEAT=490, N_STACKED_FRAMES=4)
for i in range(100):
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        # print(env.agents)
    print(f"RESET {i}")
    env.reset()
print("Passed test :)")