from env.sm64_env import SM64_ENV
from env.sm64_env_tag import SM64_ENV_TAG
from tqdm import tqdm
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            # 4 frame stack so that is the first number
            layer_init(nn.Conv2d(4, 256, 8, stride=2)),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(256, 128, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(128, 128, 2, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),

            # 7680 calculated from torch_layer_size_test.py, given 4 channels and 128x72 input
            layer_init(nn.Linear(7680, 4096)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(4096, 4096)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(4096, 2048)),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(2048,1024), std=0.01),
            nn.LeakyReLU(),
            layer_init(nn.Linear(1024,512), std=0.01),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(2048,1024), std=0.01),
            nn.LeakyReLU(),
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



# extra moves added for the more complicated model 
ACTION_BOOK = [
    # -----FORWARD
    # None
    [0,False,False,False],
    # Jump
    [0,True,False,False],
    # start longjump (crouch)
    [0,False,False,True],
    # Dive
    [0,False,True,False],

    # -----FORWARD RIGHT
    # None
    [30,False,False,False],
    [10,False,False,False],


    # -----FORWARD LEFT
    # None
    [-30,False,False,False],
    [-10,False,False,False],


    # -----BACKWARDS
    # None
    [180,False,False,False],
    # Jump
    [180,True,False,False],

    # # ----- NO STICK (no direction held)
    # # None
    # ["noStick",False,False,False],
    # # Groundpound
    # ["noStick",False,False,True],
]
env = SM64_ENV_TAG(FRAME_SKIP=4, N_RENDER_COLUMNS=4, ACTION_BOOK=ACTION_BOOK)
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

agentHider = Agent(envs).to(device)
agentHider.load_state_dict(torch.load(f"trained_models/agentHider_XL_4.pt", map_location=device))

agentSeeker = Agent(envs).to(device)
agentSeeker.load_state_dict(torch.load(f"trained_models/agentSeeker_XL_4.pt", map_location=device))


INIT_HP = {
    "MAX_EPISODES": 20,
    "MAX_EPISODE_LENGTH": 200,
}
H_S_SPLIT = env.MAX_PLAYERS//2

for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
    observations, infos = envs.reset()
    for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]), leave=False):
        
        obs_tensor = torch.Tensor(observations).to(device)
        hider_results = agentHider.get_action_and_value(obs_tensor[:H_S_SPLIT])
        seeker_results = agentSeeker.get_action_and_value(obs_tensor[H_S_SPLIT:])
        action, logprob, _, value = [torch.cat((hider_results[i], seeker_results[i])) for i in range(4)]

        observations, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
    

print("Passed test :)")