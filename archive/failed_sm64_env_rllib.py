"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from env.sm64_env import SM64_ENV

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3840, 1024)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(1024, num_outputs)
        self.value_fn = nn.Linear(1024, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = SM64_ENV(FRAME_SKIP=4, N_RENDER_COLUMNS=5)
    envs = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    envs = ss.color_reduction_v0(envs, mode="full")
    envs = ss.frame_stack_v1(envs, 4)
    # envs = ss.pettingzoo_env_to_vec_env_v1(envs)

    # env = SM64_ENV(FRAME_SKIP=4)
    # env = ss.color_reduction_v0(env, mode="B")
    # env = ss.dtype_v0(env, "float32")
    # # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    # env = ss.frame_stack_v1(env, 3)

    # env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    # env = ss.color_reduction_v0(env, mode="full")
    # # env = ss.resize_v1(env, x_size=128, y_size=72)
    # env = ss.frame_stack_v1(env, 4,stack_dim=0)
    return envs


if __name__ == "__main__":
    ray.init()

    env_name = "sm64"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=3)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="INFO",log_sys_usage=True)
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )
    c = config.to_dict()
    c["model"]["conv_filters"] = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]
    c["num_workers"] = 4
    print("-------------> ",os.environ.get("RLLIB_NUM_GPUS", "1"))
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5},
        checkpoint_freq=10,
        local_dir="J:/Github-repos/sm64-AI/rayshit/" + env_name,
        config=c,
    )