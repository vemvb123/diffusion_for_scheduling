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



def encode_image(td: TensorDict, num_groups: int, output_prefix: str, assignments: list | None = None,):

    assignment_coordinates = [[0, 0, 0] for _ in range(len(assignments))]

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
        cnode_name = f"C{i}"
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
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            pos[an] = np.array([
                x,
                y 
            ])
            print( (i,j) )
            if (i, j) in assignments:
                index = assignments.index( (i,j) )
                min_val = -5 
                max_val = 5
                x_n = (x - min_val) / (max_val - min_val)
                y_n = (y - min_val) / (max_val - min_val)
                i_n = (i - 0) / (3 - 0)
                assignment_coordinates[index] = [i_n, x_n, y_n]
            # assignments_coordinates
            # print(i, x, y)

        # Draw
        plt.figure(figsize=(8, 8))

        # Center Cnode
        plt.scatter(*pos[cnode_name], color="red", s=300)
        #plt.text(pos[cnode_name][0], pos[cnode_name][1],
        #         cnode_name, fontsize=14,
        #         ha="center", va="center")

        # Draw Anodes and labels
        for j, an in enumerate(anodes):
            g_idx, idx_in_group = anode_to_group[j]
            marker = markers[g_idx % len(markers)]
            plt.scatter(*pos[an], marker=marker, color="blue", s=150)

            # index within group just above
            y_offset = 0.025 * weights.max() * scale
            #plt.text(pos[an][0], pos[an][1] + y_offset,
            #         str(idx_in_group),
            #         fontsize=10, ha="center", va="bottom",
            #         color="green")

            # show Anode name below
            #plt.text(pos[an][0], pos[an][1] - y_offset,
            #         an, fontsize=8, ha="center", va="top")

            # weight just above the index label
            #plt.text(pos[an][0], pos[an][1] + 2 * y_offset,
            #         f"{cnode_weights[j]:.1f}",
            #         fontsize=8, ha="center", va="bottom")

            # edge line
            #xs = [pos[cnode_name][0], pos[an][0]]
            #ys = [pos[cnode_name][1], pos[an][1]]
            #plt.plot(xs, ys, color="gray", linewidth=1)

        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close()

# 2) load buffer into PIL and convert to grayscale
        buf.seek(0)
        img = Image.open(buf).convert("L")  # "L" = grayscale

# 3) save grayscale image
        filename = f"{output_prefix}_{i}.png"
        img.save(filename)

    if assignments:
        return torch.tensor(assignment_coordinates)





class ImageCoordinateDataset(Dataset):
    def __init__(self):
        # store channel sizes from the initial embeddings
        self.td = td
        self.env = env

        self.C_op = op_emb.size(1)
        self.C_ma = ma_emb.size(1)

        # store original heights (before padding) assuming width unchanged
        self.H = op_emb.size(2)

        self.op_emb = self.reshape_data(op_emb)
        self.ma_emb = self.reshape_data(ma_emb)


    def reshape_data(self, x_emb):
        x_emb = x_emb.unsqueeze(1)
        x = x_emb.detach().clone()
        x = x.squeeze(0)
        return x


    def decomposition_instance(self, sample_data):
        """
        data_instance : Tensor of shape [B, 1, C_op + C_ma, H] or [B, C_op + C_ma, H] (depending on whether
        the middle dimension 1 is kept).
        This returns:
          op_rec : shape [B, C_op, H]
          ma_rec : shape [B, C_ma, H]
        """
        # If there is a singleton dimension at dim=1 (the “1”), remove it.
        if sample_data.dim() == 4 and sample_data.size(1) == 1:
            # from [B,1,channels,H] → [B,channels,H]
            sample_data = sample_data.squeeze(1)

        # split channels
        op_rec, ma_rec = torch.split(sample_data, [self.C_op, self.C_ma], dim=1)

        # Crop spatial dimension if needed (in your case height =16 for both so likely no crop)
        op_rec = op_rec[:, :, :self.H]
        ma_rec = ma_rec[:, :, :self.H]

        return op_rec, ma_rec



    def __len__(self):
        return self.op_emb.size(0)


    def __getitem__(self, idx):
        op = self.op_emb[idx]
        ma = self.ma_emb[idx]
        
        # Pad the smaller embedding to match the larger one
        if op.shape[2] < ma.shape[2]:
            padding = (0, 0, 0, ma.shape[2] - op.shape[2])  # Pad height
            op = torch.nn.functional.pad(op, padding)
        elif ma.shape[2] < op.shape[2]:
            padding = (0, 0, 0, op.shape[2] - ma.shape[2])  # Pad height
            ma = torch.nn.functional.pad(ma, padding)
        
        # Concatenate along the channel dimension
        combined = torch.cat([op, ma], dim=1)
        
        return combined




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

    dataset_folder = 'tmp_dataset/img_coords_dataset'
    os.makedirs(dataset_folder, exist_ok=True)

    for i in range(n):
        # lag instanse
        env, td, generator_params = make_instance(4,4,4,5,20)
        # fa target fra instance 
        td_target, assignments = apply_fcfs(env, td.copy(), generator_params)
        # lage bilder og koordinater for instanse, der du kaller bilda noe spesifikt
        assignments_coordinates = encode_image(td, 4, f'img_444_{i}', assignments)
        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', assignments)
        plain_dict = tensordict_to_dict(td.copy())
        with open(f"{dataset_folder}/td_444_{i}", "w") as f:
            json.dump(plain_dict, f, indent=4)
        # lagre coords i en json, med visse navn
        torch.save(assignments_coordinates, f'{dataset_folder}/coords_444_{i}.pt')
        
        logging.info(f'Made instance {i}')







env, td, generator_params = make_instance(4,4,4,5,20)
td_fcfs, assignments = apply_fcfs(env, td.copy(), generator_params)
print(assignments)

print(td["proc_times"])
assignments_coordinates = encode_image(td, 4, 'procs', assignments)




if assignments_coordinates is not None:
    print(assignments_coordinates)
    print('')
    print(assignments)

print(assignments_coordinates[i])



make_dataset(5)



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


