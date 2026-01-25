"""
scheduling_utils.py contains all code that does scheduling
"""



import time
import random
import os
import logging
import bisect

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import torch
from tensordict import TensorDict, from_dict
from torch import Tensor

from torchtyping import TensorType
from typing import Callable, Dict, List, Tuple

from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


"""
params:
  0: number of machines
  1: ops per job
  2: number of jobs
  3: minimum proc time
  4: maximum proc time
"""

def make_instance(
    n_ma, n_jobs, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, max_eligable_ma_per_op, min_eligable_ma_per_op, batch_size
) -> Tuple[FJSPEnv, TensorDict, Dict]:

    generator_params = {
        "num_jobs": n_jobs,
        "num_machines": n_ma,
        "min_ops_per_job": min_op_per_job,
        "max_ops_per_job": max_op_per_job,
        "min_processing_time": min_proc_time,
        "max_processing_time": max_proc_time,
        "min_eligible_ma_per_op": min_eligable_ma_per_op,
        "max_eligible_ma_per_op": max_eligable_ma_per_op,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    td = env.reset(batch_size=[batch_size])
    return env, td, generator_params



def make_adj_with_order(
        actions: List, td_unscheduled: TensorDict, env: FJSPEnv, order: bool
    )-> TensorDict:

    n_actions = len(actions)

    td_to_actions = td_unscheduled.copy()
    td_to_actions = td_to_actions.unsqueeze(0)
    prev_adj = td_to_actions["ma_assignment"].clone()

    # this stores the sequence / order matrix
    assignment_adj = torch.zeros_like(prev_adj, dtype=torch.float)

    per_row_counts = torch.zeros(prev_adj.size(0), dtype=torch.int)
    for i, action in enumerate(actions):
        td_to_actions["action"] = torch.tensor([action])
        td_to_actions = env.step(td_to_actions)["next"]

        new_adj = td_to_actions["ma_assignment"]

        diff = (new_adj == 1) & (prev_adj == 0)

        if diff.any():
            normalized_order = i / float(n_actions)
            assignment_adj[diff] = normalized_order
        prev_adj = new_adj.clone()


    td_to_actions["ma_assignment"] = assignment_adj
    return td_to_actions





def make_target_orderedinput(env: FJSPEnv, td: TensorDict) -> TensorDict:

    lr_d = 1e-4
    CHECKPOINT_PATH = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_{lr_d}.ckpt"
    model = L2DModel.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td.clone(),
                    decode_type="multistart_sampling",
                    num_starts=100,
                    select_best=True,
                    return_actions=True)
    # ein kommentar
    actions = out["actions"]
    td_scheduled = make_adj_with_order(actions, td.copy(), env, 3)
    return td_scheduled, actions


# actions: [batch_size, seq_len]
def schedule_actions_batch(env: FJSPEnv, actions: List, td: TensorDict) -> TensorDict:
    for t in range(actions.size(1)):
        td["action"] = actions[:, t] 
        td = env.step(td)["next"]
    return td



def make_target(env: FJSPEnv, td: TensorDict, in_ssh: bool) -> Tuple[TensorDict, List]:

    lr_d = 1e-4

    CHECKPOINT_PATH = None
    if in_ssh: CHECKPOINT_PATH = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_{lr_d}.ckpt"
    else: CHECKPOINT_PATH = f'/home/vemund/Dokumenter/koding/d_m/rl4co_ex/rl4co_model_0.0001.ckpt'
    model = L2DModel.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]
    td_scheduled = schedule_actions_batch(env, actions, td.copy())
    return td_scheduled, actions








 

# bruk hvis ordered, for å se klart sekvens
# første operasjon er laveste tallet i return matrisen, det er annerledes enn hvordan det ellers er, der største verdi rett fra modell er første operasjon
def get_clear_sequence(assignments):
    """
    Given a tensor `assignments` with values between 0 and 1,
    return a tensor of the same shape where:
      - the largest value gets 16,
      - the 2nd largest gets 15,
      - ...
      - the 16th largest gets 1,
      - all others get 0.
    """

    flat = assignments.flatten()
    # get indices of the top 16 values
    top_vals, top_idx = torch.topk(flat, 16)

    # output tensor initialized with zeros
    out = torch.zeros_like(flat, dtype=torch.long)

    # assign values 16 → 1
    for rank, idx in enumerate(top_idx):
        out[idx] = 16 - rank

    return out.view(assignments.shape)












