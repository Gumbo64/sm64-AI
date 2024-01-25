from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from tqdm import trange
import supersuit as ss
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            # 4 frame stack so that is the first number
            layer_init(nn.Conv2d(4, 128, 8, stride=2)),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(128, 64, 4, stride=2)),
            nn.LeakyReLU(),
            nn.Flatten(),

            # 4992 calculated from torch_layer_size_test.py, given 4 channels and 128x72 input
            layer_init(nn.Linear(4992, 2048)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(1024,512), std=0.01),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(1024,512), std=0.01),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )

    def get_value(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
# env = SM64_ENV_CURIOSITY(FRAME_SKIP=4, N_RENDER_COLUMNS=4, IMG_WIDTH=480, IMG_HEIGHT=270)
env = SM64_ENV_CURIOSITY(FRAME_SKIP=4, N_RENDER_COLUMNS=4)
envs = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
envs = ss.color_reduction_v0(envs, mode="full")

envs = ss.resize_v1(envs, x_size=128, y_size=72)
envs = ss.frame_stack_v1(envs, 4)
envs = ss.black_death_v3(envs)
envs = ss.pettingzoo_env_to_vec_env_v1(envs)

envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=99999, base_class="gymnasium")
envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(envs).to(device)
agent.load_state_dict(torch.load(f"trained_models/agentCuriosity.pt", map_location=device))



INIT_HP = {
    "MAX_EPISODES": 20,
    "MAX_EPISODE_LENGTH": 200,
}

for idx_epi in trange(INIT_HP["MAX_EPISODES"]):
    observations, infos = envs.reset()
    for i in range(INIT_HP["MAX_EPISODE_LENGTH"]):
        
        obs_tensor = torch.Tensor(observations).to(device)
        results = agent.get_action_and_value(obs_tensor)
        action, logprob, _, value = results

        observations, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
    

print("Passed test :)")