from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from tqdm import tqdm
import supersuit as ss
import torch
import torch.nn as nn
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
# env = SM64_ENV_CURIOSITY(FRAME_SKIP=4, N_RENDER_COLUMNS=5, ACTION_BOOK=ACTION_BOOK, IMG_WIDTH=512,IMG_HEIGHT=288)
env = SM64_ENV_CURIOSITY(FRAME_SKIP=4, N_RENDER_COLUMNS=5, ACTION_BOOK=ACTION_BOOK)
envs = ss.black_death_v3(env)

envs = ss.resize_v1(envs, x_size=128, y_size=72)
envs = ss.clip_reward_v0(envs, lower_bound=0, upper_bound=1)
envs = ss.color_reduction_v0(envs, mode="full")
envs = ss.frame_stack_v1(envs, 1)
envs = ss.pettingzoo_env_to_vec_env_v1(envs)

envs = ss.concat_vec_envs_v1(envs, 1, num_cpus=99999, base_class="gymnasium")

envs.single_observation_space = envs.observation_space
envs.single_action_space = envs.action_space
envs.is_vector_env = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent(envs).to(device)
agent.load_state_dict(torch.load(f"trained_models/agentCuriosityLSTM_BOB.pt", map_location=device))



INIT_HP = {
    "MAX_EPISODES": 100,
    "MAX_EPISODE_LENGTH": 750,
}

next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, env.MAX_PLAYERS, agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, env.MAX_PLAYERS, agent.lstm.hidden_size).to(device),
)  

for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

    o, infos = envs.reset()
    next_obs = torch.Tensor(o).to(device)
    
    next_done = torch.zeros(env.MAX_PLAYERS).to(device)
    for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)

        observations, rewards, done, truncations, infos = envs.step(action.cpu().numpy())
        tmp = envs.step(action.cpu().numpy())
        next_obs, reward, done, truncations, infos = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

print("Finished :)")