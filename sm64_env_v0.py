from env.sm64_env import SM64_ENV

env = SM64_ENV()
# env.reset()
done = False

while not done:
    for i in range(1000000):
        actions = [0 for _ in range(env.MAX_PLAYERS)]
        if i % 10 == 0:
            actions = [2 for _ in range(env.MAX_PLAYERS)]
        if i % 10 == 1:
            actions = [1 for _ in range(env.MAX_PLAYERS)]
            

        
        # actions = [random.randint(0,env.N_ACTIONS-1) for _ in range(env.MAX_PLAYERS)]
        env.step(actions)
        env.render()
    print("RESET")
    env.reset()
