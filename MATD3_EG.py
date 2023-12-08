"""This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""
import os

import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "LR": 0.01,  # Learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
    }

    # Define the simple speaker listener environment as a parallel environment
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_episodes = 500  # Total episodes (default: 6000)
    max_steps = 25  # Maximum steps to take in each episode
    epsilon = 1.0  # Starting epsilon value
    eps_end = 0.1  # Final epsilon value
    eps_decay = 0.995  # Epsilon decay
    evo_epochs = 20  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent

    # Training loop
    for idx_epi in trange(max_episodes):
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            if INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
                    for agent_id, s in state.items()
                }

            for _ in range(max_steps):
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )

                # Get next action from agent
                cont_actions, discrete_action = agent.getAction(
                    state, epsilon, agent_mask, env_defined_actions
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                next_state, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                # Image processing if necessary for the environment
                if INIT_HP["CHANNELS_LAST"]:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                # Save experiences to replay buffer
                memory.save2memory(state, cont_actions, reward, next_state, termination)

                # Collect the reward
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(
                        agent.batch_size
                    )  # Sample replay buffer
                    agent.learn(experiences)  # Learn according to agent's RL algorithm

                # Update the state
                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    break

            # Save the total episode reward
            score = sum(agent_reward.values())
            agent.scores.append(score)

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    # Save the trained algorithm
    path = "./models/MATD3"
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.saveCheckpoint(save_path)

Watch the trained agents play

The following code allows you to load your saved MATD3 algorithm from the previous training block, test the algorithms performance, and then visualise a number of episodes as a gif.

import os

import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from PIL import Image, ImageDraw

from agilerl.algorithms.matd3 import MATD3


# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_speaker_listener_v4.parallel_env(
        continuous_actions=True, render_mode="rgb_array"
    )
    env.reset()
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Instantiate an MADDPG object
    matd3 = MATD3(
        state_dim,
        action_dim,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        device=device,
    )

    # Load the saved algorithm into the MADDPG object
    path = "./models/MATD3/MATD3_trained_agent.pt"
    matd3.loadCheckpoint(path)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 25  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = matd3.getAction(
                state,
                epsilon=0,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if matd3.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            state, reward, termination, truncation, info = env.step(action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Save the gif to specified path
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "speaker_listener.gif"), frames, duration=10
    )

