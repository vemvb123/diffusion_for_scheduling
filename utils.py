import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


import io
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional
from IPython.display import display, clear_output

# logging.info("imports")

"""
Contrains the following:
Method to vizulise a schedule
A class representing a dataset, which combined op and ma embeddings, and can also return a combined data back into a ma and op embedding
"""
# logging.info(1)
import torch
from torch.utils.data import Dataset
# logging.info("skjekk")
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.models.zoo.l2d.decoder import L2DDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.utils.trainer import RL4COTrainer
from IPython.display import display, clear_output

# logging.info(2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import time
import random
from torchvision import transforms

# logging.info(3)
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import torch

import numpy as np
import networkx as nx  # Make sure you install networkx if you don’t have it


import json

# logging.info(4)

from tensordict import TensorDict, from_dict
from torch import Tensor
from torchtyping import TensorType
from typing import Callable, Dict, List
# import inference



# logging.info("start of file")


"""
params:
  0: number of machines
  1: ops per job
  2: number of jobs
  3: minimum proc time
  4: maximum proc time
"""
def make_instance(
        ma: int, ops_per_job: int, jobs: int, min_proc: int, max_proc: int
) -> Tuple[FJSPEnv, TensorDict, Dict]:

    generator_params = {
        "num_jobs": jobs,
        "num_machines": ma,
        "min_ops_per_job": ops_per_job,
        "max_ops_per_job": ops_per_job,
        "min_processing_time": min_proc,
        "max_processing_time": max_proc,
        "min_eligible_ma_per_op": ma,
        "max_eligible_ma_per_op": ma,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    td = env.reset(batch_size=[1])
    return env, td, generator_params




def map_index_to_action(r_i, op_i, generator_params):
    job_size = generator_params["num_jobs"]
    ma_size = generator_params["num_machines"]

    return (op_i + 1) + (r_i // job_size) * ma_size


import gc
from rl4co.envs import JSSPEnv
from rl4co.models.zoo.l2d.model import L2DPPOModel
from rl4co.models.zoo.l2d.policy import L2DPolicy4PPO
from torch.utils.data import DataLoader
import json
import os

def make_target(td):
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

    actions = out["actions"]   
    return td, actions



# HERSAN
def apply_fcfs(env: FJSPEnv, td: TensorDict, generator_params: Dict
               ) -> Tuple[TensorDict, List]:

    actions = []
    ma_range = generator_params["num_machines"]

    # inneholder en liste for hver maskin, en liste inneholder sekvensen for hver op assigned til maskinen, i format [x,x]
    assignments = []
    while not td['done'].all():
        min_proc_time = 100
        action_i = 0

        rma_i = 0
        opma_i = 0
        
        ready_indexes = [i for i, val in enumerate(td['is_ready'][0]) if val]
        for r_i in ready_indexes:
            for op_i in range(ma_range):
                v = td['proc_times'][0][op_i][r_i]
                is_not_busy = td['busy_until'][0][op_i] <= td['time'][0]
                if v < min_proc_time and is_not_busy:
                    min_proc_time = v
                    action_i = map_index_to_action(r_i, op_i, generator_params)
                    # adding action taken, to assignment
                    rma_i = r_i # operasjon
                    opma_i = op_i # maskin
        

        assignments.append((opma_i, rma_i))


        actions.append(action_i)
        td['action'] = torch.tensor([action_i])
        td = env.step(td)['next']
    # print("actions during FCFS")
    # print(actions)

    return td, assignments





def save_instance():
    pass






def tensordict_to_dict(td):
    """Convert a TensorDict to a plain Python dict."""
    result = {}
    for key in td.keys():
        value = td[key]
        # if it's a tensor — convert to list
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result




def make_dataset(n):

    dataset_folder = 'data/with_targets'
    os.makedirs(dataset_folder, exist_ok=True)

    for i in range(n):
        # lag instanse
        env, td, generator_params = make_instance(4,4,4,5,50)
        # fa target fra instance 
        td_target, actions = make_target(td.copy())
        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', torch.tensor(actions))
        # lagre coords i en json, med visse navn
        torch.save(td.copy(), f'{dataset_folder}/coords_444_{i}.pt')
        
        logging.info(f'Made instance {i}')


make_dataset(100000)




