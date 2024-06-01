# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gymnasium
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from env.sm64_env_tag import SM64_ENV_TAG
from env.sm64_env_render_grid import SM64_ENV_RENDER_GRID
from tqdm import tqdm
import multiprocessing

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="sm64",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=10,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()

    # # we split batches across 2 players, so must divide this by 2
    # args.batch_size = int(args.num_envs * args.num_steps) // 2
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.image_features = nn.Sequential(
            # frame stack is the first number
            layer_init(nn.Conv2d(1, 256, 8, stride=2)),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(256, 128, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(128, 128, 2, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # do these layers just obfuscate the data? maybe just input them directly to the combined features
        # self.numerical_features = nn.Sequential(
        #     layer_init(nn.Linear(12, 512), std=0.01),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Linear(512, 512), std=0.01),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Linear(512, 512), std=0.01),
        #     nn.LeakyReLU(),
        # )

        self.combined_features_1 = nn.Sequential(
            # 7680 calculated from torch_layer_size_test.py, given 4 channels and 128x72 input
            # layer_init(nn.Linear(7680 + 512, 4096)),
            layer_init(nn.Linear(7680 + 12, 4096)),
            nn.LeakyReLU(),
        )
        self.combined_features_2 = nn.Sequential(
            layer_init(nn.Linear(4096, 4096)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(4096, 4096)),
            nn.LeakyReLU(),
        )
        self.combined_features_3 = nn.Sequential(
            layer_init(nn.Linear(4096, 4096)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(4096, 4096)),
            nn.LeakyReLU(),
        )
        self.combined_features_4 = nn.Sequential(
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

    def get_value(self, images, numerical):
        images = images.clone()
        images /= 255.0
        numerical = numerical.clone()

        image_features = self.image_features(images.permute((0, 3, 1, 2)))
        # numerical_features = self.numerical_features(numerical)
        numerical_features = numerical

        combined_1 = self.combined_features_1(torch.cat((image_features, numerical_features), dim=1))
        combined_2 = self.combined_features_2(combined_1) + combined_1
        combined_3 = self.combined_features_3(combined_2) + combined_2
        combined_4 = self.combined_features_4(combined_3)

        return self.critic(combined_4)

    def get_action_and_value(self, images, numerical, action=None):
        images = images.clone()
        images /= 255.0
        numerical = numerical.clone()

        image_features = self.image_features(images.permute((0, 3, 1, 2)))
        # numerical_features = self.numerical_features(numerical)
        numerical_features = numerical
        
        # adding +combined_1 etc is a residual connection
        combined_1 = self.combined_features_1(torch.cat((image_features, numerical_features), dim=1))
        combined_2 = self.combined_features_2(combined_1) + combined_1
        combined_3 = self.combined_features_3(combined_2) + combined_2
        combined_4 = self.combined_features_4(combined_3)

        logits = self.actor(combined_4)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(combined_4)

def combine_hider_seeker_actions(hiderActions, seekerActions):
    t = torch.zeros(2 * hiderActions.shape[0], dtype=hiderActions.dtype)
    t[0::2] = hiderActions
    t[1::2] = seekerActions
    return t

def split_hider_seeker_tensor(tensor):
    # perform the inverse of the above function
    a = tensor[::2]
    b = tensor[1::2]
    return a,b

GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
def preprocess_img(obs, space):
    # observation is (image, numerical)
    # grayscale
    new_obs = (np.expand_dims((obs[0] @ GRAYSCALE_WEIGHTS),axis=-1), obs[1])
    # print(new_obs[0].shape, new_obs[1].shape)
    return new_obs

def preprocess_space(space):
    # observation is (image, numerical)
    img_shape = space.shape
    new_img_shape = (img_shape[0], img_shape[1], 1)
    space = gymnasium.spaces.Tuple([gymnasium.spaces.Box(low=0, high=255, shape=new_img_shape, dtype=np.float32), gymnasium.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)])
    return space

if __name__ == "__main__":
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

    env = SM64_ENV_TAG(FRAME_SKIP=4, ACTION_BOOK=ACTION_BOOK, PLAYER_COLLISION_TYPE=1)

    envs = ss.observation_lambda_v0(env, preprocess_img, preprocess_space)
    envs = ss.pettingzoo_env_to_vec_env_v1(envs)
    envs.black_death = True

    num_dll = multiprocessing.cpu_count()
    # num_dll = 1

    envs = ss.concat_vec_envs_v1(envs, num_dll, num_cpus=num_dll, base_class="gymnasium")
    envs.black_death = True
    
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    args = parse_args()
    args.num_envs = envs.num_envs
    # half because it is spread across 2 players
    args.batch_size = (args.num_envs * args.num_steps) // 2
    args.minibatch_size = args.batch_size // args.num_minibatches


    run_name = f"SM64_TAG_PPO_{int(time.time())}_{env.IMG_WIDTH}x{env.IMG_HEIGHT}_PLAYERS_{env.MAX_PLAYERS}_ACTIONS_{env.N_ACTIONS}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(wandb.run.dir)
    else:
        writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    assert isinstance(envs.single_action_space, gymnasium.spaces.Discrete), "only discrete action space is supported"

    agentHider = Agent(envs).to(device)
    optimizerHider = optim.Adam(agentHider.parameters(), lr=args.learning_rate, eps=1e-5)

    agentSeeker = Agent(envs).to(device)
    optimizerSeeker = optim.Adam(agentSeeker.parameters(), lr=args.learning_rate, eps=1e-5)

    # if you want to load an agent
    # agentHider.load_state_dict(torch.load(f"trained_models/agentHider_13.1h.pt", map_location=device))
    # agentSeeker.load_state_dict(torch.load(f"trained_models/agentSeeker_13.1h.pt", map_location=device))
    
    
    # ALGO Logic: Storage setup
    # print(envs.single_observation_space)
    img_shape = envs.single_observation_space.spaces[0].shape
    numerical_shape = envs.single_observation_space.spaces[1].shape

    obsImgHider = torch.zeros((args.num_steps, args.num_envs // 2) + img_shape).to(device)
    obsNumericalHider = torch.zeros((args.num_steps, args.num_envs // 2) + numerical_shape).to(device)

    # imgsHider = 

    actionsHider = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_action_space.shape).to(device)
    logprobsHider = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    rewardsHider = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    donesHider = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    valuesHider = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)

    obsImgSeeker = torch.zeros((args.num_steps, args.num_envs // 2) + img_shape).to(device)
    obsNumericalSeeker = torch.zeros((args.num_steps, args.num_envs // 2) + numerical_shape).to(device)

    actionsSeeker = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_action_space.shape).to(device)
    logprobsSeeker = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    rewardsSeeker = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    donesSeeker = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    valuesSeeker = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # o, infoss = envs.reset()
    # next_obs = torch.Tensor(o).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    cols = 4
    renderer = SM64_ENV_RENDER_GRID(128, 72, N_RENDER_COLUMNS=cols, N_RENDER_ROWS=(envs.num_envs // cols) + 1, mode="normal")
    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizerHider.param_groups[0]["lr"] = lrnow
            optimizerSeeker.param_groups[0]["lr"] = lrnow
        obs, _ = envs.reset()
        next_obs_img = torch.Tensor(obs[0]).to(device)
        next_obs_numerical = torch.Tensor(obs[1]).to(device)

        next_done = torch.zeros(args.num_envs).to(device)
        for step in tqdm(range(0, args.num_steps), leave=False):
            global_step += 1 * args.num_envs
            # ALGO LOGIC: action logic
            with torch.no_grad():
                hider_obs_img, seeker_obs_img = split_hider_seeker_tensor(next_obs_img)
                hider_obs_numerical, seeker_obs_numerical = split_hider_seeker_tensor(next_obs_numerical)

                hider_done, seeker_done = split_hider_seeker_tensor(next_done)
                hider_results = agentHider.get_action_and_value(hider_obs_img, hider_obs_numerical)
                seeker_results = agentSeeker.get_action_and_value(seeker_obs_img, seeker_obs_numerical)
                actionHider, logprobHider, _Hider, valueHider = hider_results
                actionSeeker, logprobSeeker, _Seeker, valueSeeker = seeker_results

            obsImgHider[step], obsNumericalHider[step] = hider_obs_img, hider_obs_numerical
            obsImgSeeker[step], obsNumericalSeeker[step] = seeker_obs_img, seeker_obs_numerical

            donesHider[step], donesSeeker[step] = hider_done, seeker_done

            valuesHider[step], valuesSeeker[step] = valueHider.flatten(), valueSeeker.flatten()
            actionsHider[step], actionsSeeker[step] = actionHider, actionSeeker
            logprobsHider[step], logprobsSeeker[step] = logprobHider, logprobSeeker
            # TRY NOT TO MODIFY: execute the game and log data.

            input_actions = combine_hider_seeker_actions(actionHider, actionSeeker)
            next_obs, reward, done, truncations, infos = envs.step(input_actions.cpu().numpy())
            total_reward = torch.tensor(reward).to(device).view(-1)
            # print(total_reward)
            # time.sleep(0.1)
            rewardsHider[step], rewardsSeeker[step] = split_hider_seeker_tensor(total_reward)

            # need to input the numpy form of the observation to the renderer, not tensor
            renderer.render_game(next_obs)
            next_obs_img = torch.Tensor(next_obs[0]).to(device)
            next_obs_numerical = torch.Tensor(next_obs[1]).to(device)
            next_done = torch.Tensor(done).to(device)
            


            # for idx, item in enumerate(info):
            #     player_idx = idx % 2
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, {player_idx}-episodic_return={item['episode']['r']}")
            #         writer.add_scalar(f"charts/episodic_return-player{player_idx}", item["episode"]["r"], global_step)
            #         writer.add_scalar(f"charts/episodic_length-player{player_idx}", item["episode"]["l"], global_step)

        # bootstrap value if not done
        hider_obs_img, seeker_obs_img = split_hider_seeker_tensor(next_obs_img)
        hider_obs_numerical, seeker_obs_numerical = split_hider_seeker_tensor(next_obs_numerical)
        with torch.no_grad():
            ############### HIDER BOOTSTRAPPING ###############
            hider_values = agentHider.get_value(hider_obs_img, hider_obs_numerical)
            next_value = hider_values.reshape(1, -1)
            advantagesHider = torch.zeros_like(rewardsHider).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - hider_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - donesHider[t + 1]
                    nextvalues = valuesHider[t + 1]
                delta = rewardsHider[t] + args.gamma * nextvalues * nextnonterminal - valuesHider[t]
                advantagesHider[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returnsHider = advantagesHider + valuesHider

            ############### SEEKER BOOTSTRAPPING ###############
            seeker_values = agentSeeker.get_value(seeker_obs_img, seeker_obs_numerical)
            next_value = seeker_values.reshape(1, -1)
            advantagesSeeker = torch.zeros_like(rewardsSeeker).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - seeker_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - donesSeeker[t + 1]
                    nextvalues = valuesSeeker[t + 1]
                delta = rewardsSeeker[t] + args.gamma * nextvalues * nextnonterminal - valuesSeeker[t]
                advantagesSeeker[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returnsSeeker = advantagesSeeker + valuesSeeker


        ################### HIDER LEARNING ###################

        # flatten the batch
        b_img_obs = obsImgHider.reshape((-1,) + img_shape)
        b_numerical_obs = obsNumericalHider.reshape((-1,) + numerical_shape)

        b_logprobs = logprobsHider.reshape(-1)
        b_actions = actionsHider.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantagesHider.reshape(-1)
        b_returns = returnsHider.reshape(-1)
        b_values = valuesHider.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agentHider.get_action_and_value(b_img_obs[mb_inds], b_numerical_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizerHider.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agentHider.parameters(), args.max_grad_norm)
                optimizerHider.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizerHider.param_groups[0]["lr"], global_step)
        writer.add_scalar("lossesHider/value_loss", v_loss.item(), global_step)
        writer.add_scalar("lossesHider/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("lossesHider/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("lossesHider/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("lossesHider/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("lossesHider/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("lossesHider/explained_variance", explained_var, global_step)

        ################### SEEKER LEARNING (should be the same code) ###################
                 
        # flatten the batch
        b_img_obs = obsImgSeeker.reshape((-1,) + img_shape)
        b_numerical_obs = obsNumericalSeeker.reshape((-1,) + numerical_shape)

        b_logprobs = logprobsSeeker.reshape(-1)
        b_actions = actionsSeeker.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantagesSeeker.reshape(-1)
        b_returns = returnsSeeker.reshape(-1)
        b_values = valuesSeeker.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agentSeeker.get_action_and_value(b_img_obs[mb_inds], b_numerical_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizerSeeker.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agentSeeker.parameters(), args.max_grad_norm)
                optimizerSeeker.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizerHider.param_groups[0]["lr"], global_step)
        writer.add_scalar("lossesSeeker/value_loss", v_loss.item(), global_step)
        writer.add_scalar("lossesSeeker/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("lossesSeeker/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("lossesSeeker/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("lossesSeeker/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("lossesSeeker/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("lossesSeeker/explained_variance", explained_var, global_step)

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # average over time and envs
        writer.add_scalar("charts/avg_hider_reward", torch.mean(torch.mean(rewardsHider)), global_step)
        writer.add_scalar("charts/avg_seeker_reward", torch.mean(torch.mean(rewardsSeeker)), global_step)

        if args.track:
            # make sure to tune `CHECKPOINT_FREQUENCY` 
            # so models are not saved too frequently
            if update % 360 == 0:
                t = time.time() - start_time
                t_hours = round(t / 3600, 1)
                torch.save(agentHider.state_dict(), f"{wandb.run.dir}/agentHider_{t_hours}h.pt")
                torch.save(agentSeeker.state_dict(), f"{wandb.run.dir}/agentSeeker_{t_hours}h.pt")
                wandb.save(f"{wandb.run.dir}/agentHider.pt", policy="now",  base_path=wandb.run.dir)
                wandb.save(f"{wandb.run.dir}/agentSeeker.pt", policy="now",  base_path=wandb.run.dir)

    envs.close()
    writer.close()
