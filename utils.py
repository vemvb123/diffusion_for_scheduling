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




# weights = td["proc_times"][0]  # shape (2, 4)
def encode_image(td: TensorDict, num_groups: int, output_prefix: str, assignments: list | None = None,):
   # weights shape: (num_cnodes, num_anodes)

    weights = td['proc_times'][0]
    num_cnodes, num_anodes = weights.shape

    # Anode names
    anodes = [f"A{j}" for j in range(num_anodes)]

    # ----- Compute contiguous groups -----
    base = num_anodes // num_groups
    extra = num_anodes % num_groups
    group_ranges = []
    start = 0
    for g in range(num_groups):
        size = base + (1 if g < extra else 0)
        group_ranges.append((start, start + size))
        start += size

    # Map anode index -> (group_index, index_in_group)
    anode_to_group = {}
    for g_idx, (s, e) in enumerate(group_ranges):
        for idx_in_group, a_idx in enumerate(range(s, e)):
            anode_to_group[a_idx] = (g_idx, idx_in_group)

    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    
    # For each Cnode → one image
    for i in range(num_cnodes):
        cnode_name = f"{i}"
        cnode_weights = weights[i]

        pos = {}
        # Center Cnode
        pos[cnode_name] = np.array([0.0, 0.0])

        # Place Anodes
        angle_step = 2 * np.pi / num_anodes
        scale = 0.25  # scale factor so plot doesn’t blow up
        for j, an in enumerate(anodes):
            # **distance proportional to weight**
            radius = scale * cnode_weights[j]
            theta = j * angle_step
            pos[an] = np.array([
                radius * np.cos(theta),
                radius * np.sin(theta)
            ])
        
        


        # Draw
        plt.figure(figsize=(8, 8))

        # Center Cnode
        plt.scatter(*pos[cnode_name], color="red", s=300)
        plt.text(pos[cnode_name][0], pos[cnode_name][1],
                 cnode_name, fontsize=14,
                 ha="center", va="center")

        # Draw Anodes and labels
        for j, an in enumerate(anodes):
            g_idx, idx_in_group = anode_to_group[j]
            marker = markers[g_idx % len(markers)]
            plt.scatter(*pos[an], marker=marker, color="blue", s=150)

            # index within group just above
            y_offset = 0.025 * weights.max() * scale
            plt.text(pos[an][0], pos[an][1] + y_offset,
                     str(idx_in_group),
                     fontsize=10, ha="center", va="bottom",
                     color="green")

            # show Anode name below
            # plt.text(pos[an][0], pos[an][1] - y_offset, an, fontsize=8, ha="center", va="top")

            # weight just above the index label
            # plt.text(pos[an][0], pos[an][1] + 2 * y_offset,
            #          f"{cnode_weights[j]:.1f}",
            #          fontsize=8, ha="center", va="bottom")
            
            # edge line
            # xs = [pos[cnode_name][0], pos[an][0]]
            # ys = [pos[cnode_name][1], pos[an][1]]
            # plt.plot(xs, ys, color="gray", linewidth=1)



        plt.axis("off")
        # plt.title(f"Cnode {cnode_name} and its Anodes (dist ∝ weight)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)  # save to in‑memory buffer
        buf.seek(0)

        img = Image.open(buf).convert("L")  # "L" mode = 8‑bit grayscale :contentReference[oaicite:1]{index=1}

        filename = f"{output_prefix}_{i}.png"
        img.save(filename)

        plt.close()

        print(f"Image saved: {filename}")

    if assignments:
        return assignments_coordinates





env, td, generator_params = make_instance(4,4,4,5,20)

td, assignments = apply_fcfs(env, td, generator_params)
print(assignments)

# print(td["proc_times"])
assignments_coordinates = encode_image(td, 4, 'procs', assignments)
print(assignments_coordinates)



"""


I have four points, call these Cnodes.
These four points should be in the center, with a bit of distance between them.
Then I have this matrix:

    proc_times

each column annotes a Cnode's relation to the Anodes (nodes of another type).
There is a row for each Anode. 
The number for some Cnode-Anode, represents the weight between the Anode and the Cnode.

The Cnodes are plotted in the center.
We encode the weight as distance between Anodes and Cnodes.
Now, make code for plotting the Anodes and Cnodes. 
Show product as an image


"""


