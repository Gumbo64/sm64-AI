
from env.sm64_env_rrt import SM64_ENV_RRT
from tqdm import trange
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
            layer_init(nn.Conv2d(1, 256, 8, stride=2)),
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
        )
        self.lstm = nn.LSTM(4096, 2048)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

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

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        x = x.clone()
        hidden, _ = self.get_states(x.permute((0, 3, 1, 2)), lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        # the /255 is done in get_states
        x = x.clone()
        hidden, lstm_state = self.get_states(x.permute((0, 3, 1, 2)), lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state



# env = SM64_ENV_TAG(FRAME_SKIP=4, N_RENDER_COLUMNS=4, IMG_WIDTH=480, IMG_HEIGHT=270)
    
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
    # Jump
    # [30,True,False,False],

    # -----FORWARD LEFT
    # None
    [-30,False,False,False],
    [-10,False,False,False],
    # Jump
    # [-30,True,False,False],

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

env = SM64_ENV_RRT(FRAME_SKIP=4, N_RENDER_COLUMNS=4, ACTION_BOOK=ACTION_BOOK, NODE_RADIUS= 400, LOAD_PATH="rrt_stuff" )
envs = ss.clip_reward_v0(env, lower_bound=0, upper_bound=1)
envs = ss.color_reduction_v0(envs, mode="full")

envs = ss.resize_v1(envs, x_size=128, y_size=72)
envs = ss.frame_stack_v1(envs, 1)
envs = ss.black_death_v3(envs)
envs = ss.pettingzoo_env_to_vec_env_v1(envs)

envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=99999, base_class="gymnasium")
envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


agent = Agent(envs).to(device)
agent.load_state_dict(torch.load(f"trained_models/agentRRT.pt", map_location=device))


INIT_HP = {
    "MAX_EPISODES": 10,
    "MAX_EPISODE_LENGTH": 200,
    "MAX_PLAYERS":env.MAX_PLAYERS,
}


next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, INIT_HP["MAX_PLAYERS"], agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, INIT_HP["MAX_PLAYERS"], agent.lstm.hidden_size).to(device),
)  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
for idx_epi in trange(INIT_HP["MAX_EPISODES"]):
    observations, infos = envs.reset()
    done_tensor = torch.zeros(INIT_HP["MAX_PLAYERS"]).to(device)
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
    for i in range(INIT_HP["MAX_EPISODE_LENGTH"]):

        obs_tensor = torch.Tensor(observations).to(device)
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(obs_tensor, next_lstm_state, done_tensor)
        observations, rewards, done, truncations, infos = envs.step(action.cpu().numpy())
        done_tensor = torch.Tensor(done).to(device)

    # envs.render()
    

print("Passed test :)")