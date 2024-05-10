# from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from env.sm64_env_curiosity import SM64_ENV_CURIOSITY
from tqdm import tqdm
import supersuit as ss
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID
# import multiprocessing

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
if __name__ == "__main__":
    INIT_HP = {
        "MAX_EPISODES": 1,
        "MAX_EPISODE_LENGTH": 3000,
        "N_RENDER_COLUMNS": 5,
        "IMAGE_SCALE_FACTOR": 1,
    }
        
    env = SM64_ENV_CURIOSITY(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK,
                                NODES_MAX=3000, NODE_RADIUS= 400, NODES_MAX_VISITS=400, NODE_MAX_HEIGHT_ABOVE_GROUND=1000,
                                MAKE_OTHER_PLAYERS_INVISIBLE=False, IMG_WIDTH=128 * INIT_HP["IMAGE_SCALE_FACTOR"], IMG_HEIGHT=72 * INIT_HP["IMAGE_SCALE_FACTOR"])
    envs = ss.black_death_v3(env)
    envs = ss.clip_reward_v0(envs, lower_bound=-1, upper_bound=1)
    # envs = ss.color_reduction_v0(envs, mode="full")
    envs = ss.frame_stack_v1(envs, 1)
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)

    # num_dll = multiprocessing.cpu_count()
    num_dll = 1

    envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    # envs.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(f"trained_models/agentCuriosityLSTM_BOB.pt", map_location=device))

    GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    def preprocess_img(obs):
        # grayscale
        new_obs = np.expand_dims((obs @ GRAYSCALE_WEIGHTS),axis=-1)
        # downscale
        return new_obs[:,::INIT_HP["IMAGE_SCALE_FACTOR"],::INIT_HP["IMAGE_SCALE_FACTOR"],:]
    
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, envs.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, envs.num_envs, agent.lstm.hidden_size).to(device),
    )  
    
    renderer = SM64_ENV_RENDER_GRID(128 * INIT_HP["IMAGE_SCALE_FACTOR"], 72 * INIT_HP["IMAGE_SCALE_FACTOR"], N_RENDER_COLUMNS=INIT_HP["N_RENDER_COLUMNS"], N_RENDER_ROWS=envs.num_envs / INIT_HP["N_RENDER_COLUMNS"], coloured=True, mode="normal")
    for idx_epi in tqdm(range(INIT_HP["MAX_EPISODES"])):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        o, infos = envs.reset()
        next_obs = torch.Tensor(preprocess_img(o)).to(device)
        next_done = torch.zeros(envs.num_envs).to(device)
        for i in tqdm(range(INIT_HP["MAX_EPISODE_LENGTH"]),leave=False):
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
            observations, rewards, done, truncations, infos = envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(preprocess_img(observations)).to(device), torch.Tensor(done).to(device)
            renderer.render_game(observations)
    envs.reset()
    print("Finished :)")