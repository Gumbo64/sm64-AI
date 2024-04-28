import optuna
import time
from cleanrl_utils.tuner import Tuner
# # HAS ERROR
# Traceback (most recent call last):
#   File "J:\Github-repos\sm64-AI\sm64-AI\lib\site-packages\optuna\study\_optimize.py", line 200, in _run_trial
#     value_or_values = func(trial)
#   File "J:\Github-repos\sm64-AI\sm64-AI\lib\site-packages\cleanrl_utils\tuner.py", line 92, in objective
#     experiment = runpy.run_path(path_name=self.script, run_name="__main__")
#   File "J:\devtools\Python\lib\runpy.py", line 288, in run_path
#     return _run_module_code(code, init_globals, run_name,
#   File "J:\devtools\Python\lib\runpy.py", line 97, in _run_module_code
#     _run_code(code, mod_globals, init_globals,
#   File "J:\devtools\Python\lib\runpy.py", line 87, in _run_code
#     exec(code, run_globals)
#   File "sm64_env_cleanrl.py", line 268, in <module>
#     _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
#   File "sm64_env_cleanrl.py", line 121, in get_action_and_value
#     probs = Categorical(logits=logits)
#   File "J:\Github-repos\sm64-AI\sm64-AI\lib\site-packages\torch\distributions\categorical.py", line 70, in __init__
#     super().__init__(batch_shape, validate_args=validate_args)
#   File "J:\Github-repos\sm64-AI\sm64-AI\lib\site-packages\torch\distributions\distribution.py", line 68, in __init__
#     raise ValueError(
# ValueError: Expected parameter logits (Tensor of shape (31, 10)) of distribution Categorical(logits: torch.Size([31, 10])) to 
# satisfy the constraint IndependentConstraint(Real(), 1), but found invalid values:
# tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
#         [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],



wandb_args = {
    "project":"sm64",
    "sync_tensorboard":True,
    # config=vars(args),
    # "name":f"SM64_PPO_{int(time.time())}",
    "monitor_gym":True,
    # "save_code":True,
}

tuner = Tuner(
    script="sm64_env_cleanrl.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "sm64": None,
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.003, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4, 8] ),
        "update-epochs": trial.suggest_categorical("update-epochs", [1, 2, 4, 8]),
        "num-steps": trial.suggest_categorical("num-steps", [100, 125, 150, 175, 200]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
        "total-timesteps": 100000,
        "num-envs": 1,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
    wandb_kwargs=wandb_args,
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)