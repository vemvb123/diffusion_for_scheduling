import logging

logging.info(1)
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


logging.info(2)

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

logging.info(3)
import torch
from torch.utils.data import Dataset
# logging.info("skjekk")
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel

# from rl4co.models.zoo.l2d.policy import L2DPolicy
# from rl4co.models.zoo.l2d.decoder import L2DDecoder
# from rl4co.models.nn.graph.hgnn import HetGNNEncoder
# from rl4co.utils.trainer import RL4COTrainer
# from IPython.display import display, clear_output

# logging.info(2)

logging.info(4)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import time
import random
from torchvision import transforms

# logging.info(3)
from torch.utils.data import Dataset
# from PIL import Image
import glob
import os
import torch

import numpy as np
import networkx as nx  # Make sure you install networkx if you don’t have it


logging.info(5)
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
        ma: int, ops_per_job: int, jobs: int, min_proc: int, max_proc: int, batch_size: int
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
    td = env.reset(batch_size=[batch_size])
    return env, td, generator_params




def map_index_to_action(r_i, op_i, generator_params):
    job_size = generator_params["num_jobs"]
    ma_size = generator_params["num_machines"]

    return (op_i + 1) + (r_i // job_size) * ma_size


logging.info(6)
import gc
from rl4co.envs import JSSPEnv
from rl4co.models.zoo.l2d.model import L2DPPOModel
from rl4co.models.zoo.l2d.policy import L2DPolicy4PPO
from torch.utils.data import DataLoader
import json
import os


# 1: no order
# 2: ordered


# order .. første skedulerte op har minst verdi, sist skedulerte op har størst verdi
def schedule_actions(
        actions, td_unscheduled, env, order: bool
):
    """
    ordered == 2: global normalized order (i / n_actions)
    ordered == 3: per-row normalized order
    ordered == 1: (reserved / undefined – keep same as 1 for now)
    """

    n_actions = len(actions)

    td_to_actions = td_unscheduled.copy()
    td_to_actions = td_to_actions.unsqueeze(0)
    prev_adj = td_to_actions["ma_assignment"].clone()

    # this stores the sequence / order matrix
    assignment_adj = torch.zeros_like(prev_adj, dtype=torch.float)

    # only used for ordered == 2
    # count how many assignments so far per row
    per_row_counts = torch.zeros(prev_adj.size(0), dtype=torch.int)
    for i, action in enumerate(actions):
        td_to_actions["action"] = torch.tensor([action])
        td_to_actions = env.step(td_to_actions)["next"]

        new_adj = td_to_actions["ma_assignment"]

        diff = (new_adj == 1) & (prev_adj == 0)

        if diff.any():

            # global normalized order
            normalized_order = i / float(n_actions)
            assignment_adj[diff] = normalized_order
        prev_adj = new_adj.clone()


    td_to_actions["ma_assignment"] = assignment_adj
    # td_to_actions = td_to_actions.squeeze(0)
    return td_to_actions





def make_target_orderedinput(env, td: TensorDict, ordered: int):
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
    td_scheduled = schedule_actions(actions, td.copy(), env, 3)
    return td_scheduled, actions




"""
def schedule_actions(env, actions, td):
    for action in actions[0]:
        td['action'] = torch.tensor([action])
        td = env.step(td)['next']
    return td
"""


def schedule_actions_batch(env, actions, td):
    # actions: [batch_size, seq_len]
    for t in range(actions.size(1)):
        # take the batch of actions at time t
        td["action"] = actions[:, t]   # shape [batch_size]
        td = env.step(td)["next"]
    return td



def make_target(env, td, in_ssh):
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
    print("Making dataset...")
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'
    os.makedirs(dataset_folder, exist_ok=True)
    batch_size = 184
    for i in range(0, n, batch_size):
        # lag instanse
        env, td, generator_params = make_instance(4,4,4,5,50, batch_size)
        # fa target fra instance 
        td_target, actions = make_target(env, td.copy(), True)
        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', torch.tensor(actions))
        # lagre coords i en json, med visse navn
        torch.save(td.copy(), f'{dataset_folder}/{i}_{i+batch_size}.pt')
        logging.info(f'Made instance {i} to {i+batch_size}')

    logging.info(f'Made all {i+batch_size} instances. Done making dataset')

# make_dataset(20000)
# exit()

# returnerer matriser i riktig shape, som inneholder featursene
# ting utenfor adj blir maskert med 1
# nar loss kalkuleres, fjernes ting utenfor rammen for instanse
# lager adj matriser for features som ikke gis av instanse ogsa, ex jobb tilhorer osv..
#..
# nar lager datasett, loader td fra fil
# sa sender td inn til denne,
# ma ogsa returneree target
# normaliserer

logging.info(7)
import torch.nn.functional as F

def expand_matrix(x: torch.Tensor, shape_to_make: tuple[int, int], max_and_min: tuple[int, int]) -> torch.Tensor:
    min_val = max_and_min[0]
    max_val = max_and_min[1]
    x_norm = (x - min_val) / (max_val - min_val)
    x_norm = x_norm.clamp(0, 1)  # ensure range [0,1]

    h = None
    w = None
    if len(x_norm.shape) == 2:
        h, w = x_norm.shape
    else:
        _, h, w = x_norm.shape

    pad_bottom = shape_to_make[0] - h
    pad_right = shape_to_make[1] - w
    x_padded = F.pad(x_norm, (0, pad_right, 0, pad_bottom), value=0.0)

    # x_final = x_padded.unsqueeze(1)
    return x_padded


# TODO
# trene med forskjelige loss verdier...
# trener da for 2 epoker, så ser hvilken av lossene som er minst, og trener en modell for flere epoker med den lr
def get_feature_adj_from_instance(td: TensorDict, env, order: bool) -> tuple[
        torch.Tensor, # target assignments
        torch.Tensor, # proc times matrix
        torch.Tensor, # jobid matrix
        torch.Tensor # pos in job matrix
        ]:
    # bytt senere ut med assignments fra target
    assignments = None
    if order:
        td_scheduled = schedule_actions(td['opt_actions'], td.copy(), env, order)
        assignments = td_scheduled['ma_assignment']
        # assignments = assignments.unsqueeze(0)
    else:
        assignments = td['opt_assignment']
        assignments = assignments.unsqueeze(0)

    assignments = expand_matrix(assignments, (20, 20), (0, 1))

    proc_times = td['proc_times']
    proc_times = proc_times.unsqueeze(0)
    proc_times = expand_matrix(proc_times, (20,20), (5,50))

    # TODO endre hvis annerledes jobber
    n_jobs = 4

    # matrix for jobid
    job_id = td['ops_job_map']
    job_id = job_id.repeat(16, 1).unsqueeze(0)  # now shape is (1, 16, 16)
    job_id = expand_matrix(job_id, (20,20), (0,n_jobs))


    # pos in job matrix
    pos_job = torch.tensor([0, 1, 2, 3], dtype=torch.float)
    pos_job = pos_job.repeat(4)  # shape (16,)
    pos_job = pos_job.unsqueeze(0).repeat(16, 1)  # shape (16,16)
    pos_job = pos_job.unsqueeze(0)
    pos_job = expand_matrix(pos_job, (20,20), (0,n_jobs))

    """
    logging.info("shapes from get_feature_adj_from_instance:")
    logging.info(assignments.shape)
    logging.info(proc_times.shape)
    logging.info(job_id.shape)
    logging.info(pos_job.shape)
    """

    return assignments, proc_times, job_id, pos_job
'''
jobs = 4
ma = 4
ops_per_job = 4
max_proc = 50
min_proc = 5

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

get_feature_adj_from_instance(td)
'''

logging.info(8)
import os
import torch
from torch.utils.data import Dataset


"""
class Dataset_RL4CO(Dataset):
    def __init__(self, folder, ordered: bool, generator_params, transform=None):
        self.folder = folder
        self.transform = transform
        self.ordered = ordered
        self.generator_params = generator_params
        self.env = FJSPEnv(generator_params=self.generator_params)

        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".pt")
        ]
        self.files.sort()  # optional but often helpful

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # load the TensorDict
        td = torch.load(file_path)
        target_assignments, proc_times, job_id, pos_job = get_feature_adj_from_instance(td, self.env, self.ordered)

        if self.transform:
            tensordict = self.transform(td)

        return target_assignments, proc_times, job_id, pos_job
"""

logging.info(9)
import bisect

class Dataset_RL4CO(Dataset):
    def __init__(self, folder, generator_params, order,transform=None):
        self.folder = folder
        self.transform = transform
        self.order = order
        self.generator_params = generator_params
        self.env = FJSPEnv(generator_params=self.generator_params)

        self.files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".pt")
        )

        # Precompute number of instances per file
        self.file_sizes = []
        for f in self.files:
            td = torch.load(f, map_location="cpu", weights_only=False)
            self.file_sizes.append(self._get_batch_size(td))

        # Prefix sum for fast index lookup
        self.cum_sizes = [0]
        for size in self.file_sizes:
            self.cum_sizes.append(self.cum_sizes[-1] + size)

    def _get_batch_size(self, td):
        """
        Infer batch size from TensorDict or dict of tensors.
        """
        # Example for TensorDict
        return td.batch_size[0]
        # or, if plain dict:
        # return next(iter(td.values())).shape[0]

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        # Find which file this idx belongs to
        file_idx = bisect.bisect_right(self.cum_sizes, idx) - 1
        instance_idx = idx - self.cum_sizes[file_idx]

        file_path = self.files[file_idx]
        td = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        # Select a single instance from the batch
        td_instance = td[instance_idx]

        if self.transform:
            td_instance = self.transform(td_instance)

        target_assignments, proc_times, job_id, pos_job = \
            get_feature_adj_from_instance(
                td_instance, self.env, self.order
            )

        for data in [target_assignments, proc_times, job_id, pos_job]:
            if torch.isnan(data).any():
                raise ValueError("Assignment NaN values found in tensor")

        return target_assignments, proc_times, job_id, pos_job


logging.info(10)

from inference import feature_inference, adj_inference, guide_adj_inference

# første skedulerte har minst verdi, sist skedulerte har størst verdi
def show_order_clear(x, n_values):

    print(x)
    print("--")    
    # flatten all values
    flat = x.flatten()

    # find the top 16 values and their indices
    topk_vals, topk_idx = torch.topk(flat, n_values)

    # sort those top 16 in descending order so largest -> rank 1
    sorted_vals, sorted_order = torch.sort(topk_vals, descending=False)
    top16_idx_sorted = topk_idx[sorted_order]

    # create an output tensor of zeros
    out = torch.zeros_like(flat)

    # assign ranks 1..16 to those positions
    for rank, idx in enumerate(top16_idx_sorted, start=1):
        out[idx] = rank

    # reshape back to original
    return out.view_as(x)





def round_to_values(x: torch.Tensor, n_values: int) -> torch.Tensor:
    orig_shape = x.shape

    # assume shape [1,1,4,16] or similar, so flatten leading dims
    flat = x.view(-1, x.shape[-2], x.shape[-1])  # [B, 4, 16]

    # find max in each column along row dim (dim=1)
    max_vals, _ = flat.max(dim=1, keepdim=True)  # [B, 1, 16]

    # compare with max and binarize
    mask = torch.isclose(flat, max_vals)  # True where value == max

    # convert to float (1.0/0.0)
    result = mask.float()

    # restore original leading dims
    result = result.view(orig_shape)

    print(result)

    return result
    """
    # flatten and get top k indices
    topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

    # start with all zeros
    y = torch.zeros_like(x).flatten()

    # set top entries to 1
    y[topk_idx] = 1.0

    # reshape back to original shape
    return y.view(x.shape)

    """



#def map_assignemnts_to_actions(assignments, ordered:bool):
#    actions = []
#    if ordered:
        # assignments matrise
        # er i 4 seksjoner av matrisen, 4columns etter 4columns, helt til 16 kolonner, delt i 4 seksjoner
        # henter største verdi, ser hvilken seksjon den er innenfor
        # ser hvilken rad det ligger på. første rad er 1, andre rad har verdi 2, osv.
        # de er delt inn i de fire seksjonene. 1 rad innenfor 1st seksjon er 1, 3rd rad innenfor 2nd seksjon er 4*2+3=11, osv...
        # kartlegger alle disse verdiene inn i en liste. Først den størst verdien (eks 0.87) i matrisen sin verdi (eks kanskje mappes 14), 
        # så neste største verdi (eks 0.79), mappes til 18
        # så får man etterhvert en liste på 16 tall [14,18 ....]



def map_assignemnts_to_actions(assignments, order: bool):

    # remove batch/channel dims if present
    if assignments.dim() == 4:
        assignments = assignments.squeeze(0).squeeze(0)  # (4,16)

    H, W = assignments.shape  # H=4, W=16
    section_width = 4
    num_sections = W // section_width
    """
    actions = []
    if order:
        x = assignments

        x = x.squeeze(0).squeeze(0)

        # get all nonzero positions
        rows, cols = torch.nonzero(x, as_tuple=True)

        # get the values at those positions
        vals = x[rows, cols]

        # sort by value (1 → 16)
        order = torch.argsort(vals)
        rows = rows[order]
        cols = cols[order]

        # compute mapped values
        sections = cols // 4
        mapped = sections * 4 + (rows + 1)

        return mapped.tolist()


    # TODO inkluder dette igjen i funksjonen seinere
    """
    if order:
        # order by largest value first
        _, indices = torch.topk(assignments.flatten(), H * W)

        for idx in indices:
            row = idx // W
            col = idx % W
            section = col // section_width
            action = section * H + row + 1
            actions.append(action.item())

    else:
        cols_per_section = 4
        actions = []

        for col in range(assignments.shape[1]):
            section_idx = col // cols_per_section
            for row in range(assignments.shape[0]):
                if assignments[row, col] == 1: 
                    value = section_idx * cols_per_section + (row + 1)
                    actions.append(value)

        return actions


def make_step(env, td, action):
    td['action'] = torch.tensor([action])
    td = env.step(td)['next']


    return td


import time
import random



def inferenced_schedule(assignments, order: bool, env, td, path_save_image: str):
    actions = map_assignemnts_to_actions(assignments, order)
    print(actions)
    # print(assignments)
    # print(td["opt_assignment"])
    # print(actions)
    # print(td["opt_actions"])
    # exit()
    td.del_("opt_assignment")
    td.del_("opt_actions")

    #print(td.shape)
    #td = TensorDict.from_dict(td, auto_batch_size=True)
    #print(td.shape)
    #td = TensorDict(td, batch_size=[1])
    #print(td.shape)

    td = td.unsqueeze(0)
    print("scheduling actions")
    env.render(td, 0)


    #---
    while not td["done"].all():
        # loop true indexer
        # for en true index, se at maskinen den oppgaven skal skeduleres til ikke er opptatt
        # det ses ved at: mapped = [((v - 1) % 4) + 1 for v in actions]  ... fra ctions
        # hvis opptatt, gå til neste gå til neste true.
        # så den oppgaven endelig kn skeduleres, skeduleres den, så starter du å loope true fra starten av
        # time.sleep(10)
        #print(td["time"])
        #print(td["busy_until"])
        #print(td["is_ready"])
        #print(assignments)

        if order:
            pass

        else:
            ready_ops = torch.nonzero(td["is_ready"], as_tuple=True)[1]

            ma_indices_for_actions = [((v - 1) % 4) for v in actions]  # 0‑based
            # print("machine indices:", ma_indices_for_actions)

            for op in ready_ops:
                op = op.item()
                machine_idx = ma_indices_for_actions[op]
                busy_val = td["busy_until"][0, machine_idx]
                current_time = td["time"][0]

                if busy_val <= current_time:
                    td['action'] = torch.tensor([actions[op]])
                    td = env.step(td)['next']
                    env.render(td, 0)
                    print(f"did action {actions[op]}")
                    print(f"actions: {actions}")
                else:
                    td["time"] = busy_val.unsqueeze(0)
                    print("could not schedule, jumped in time")

    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
    return td



 
# /cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444/0_184.pt

def get_td_from_path(path: str, instance_idx: int) -> Tensor:
    # Get sorted list of data files
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    # Build file ranges
    cum_sizes = []
    ranges = []
    total = 0

    for fname in files:
        start, end = map(int, fname.replace(".pt", "").split("_"))
        size = end - start
        cum_sizes.append(total)
        ranges.append((start, end))
        total += size

    # Find which file contains the instance
    file_idx = bisect.bisect_right(cum_sizes, instance_idx) - 1
    if file_idx < 0:
        raise ValueError(f"Instance {instance_idx} not found in {path}")

    file_path = os.path.join(path, files[file_idx])

    # Load with weights_only=False so that TensorDict objects (or other custom objects)
    # can be unpickled properly. Only do this if the file is from a trusted source.
    batch = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,  # use full pickle, not restricted weights_only loader
    )

    # Compute local index within this batch
    local_idx = instance_idx - cum_sizes[file_idx]
    return batch[local_idx]


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




"""
adj_1_loss_over_epochs.png  adj_type_4.pth  enc_type_4.pth            f_2_loss_over_epochs.png  model_adj_adj_noorder.pth      model_feature_adj_order.pth
adj_2_loss_over_epochs.png  enc_type_1.pth  enc_type_5.pth            f_4_loss_over_epochs.png  model_adj_adj_order.pth        model_feature_enc_noorder.pth
adj_type_1.pth              enc_type_2.pth  f_1_loss_over_epochs.png  f_5_loss_over_epochs.png  model_feature_adj_noorder.pth  model_feature_enc_order.pth
"""


# 4 features
# 5 features ordered
# 1 adj
# 2 adj ordered

def get_inference_result(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = get_td_from_path(dataset_folder, instance_idx)

    env, td_ignore, generator_params = make_instance(4,4,4,5,50, batch_size=1)

    target_assignments, proc_times, job_id, pos_job = get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")

    inference_assignments = None
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = adj_inference(proc_times, job_id, pos_job, adj_model_path, n_samples)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print("Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :4, :16]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = show_order_clear(inference_assignments, 16)
    else:
        inference_assignments = round_to_values(inference_assignments, 16)


    #print("inferenced results here:")
    # target_assignments = target_assignments[:,:,:4,:16]
    #print(inference_assignments)
    #print(target_assignments)
    # target_assignments = show_order_clear(target_assignments, 16)

    # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    """
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :4, :16]
        if order:
            assignment_at_time = round_to_values(assignment_at_time, 16)
        else:
            assignment_at_time = show_order_clear(assignment_at_time, 16)

        if torch.equal(assignment_at_time, inference_assignments):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(inference_assignments)
            break
    """

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    # TODO fjern
    # td_scheduled = inferenced_schedule(target_assignments, order, env, td.copy(), graph_save_path)
    # TODO gjør seinere så ikke kommenter ut
    td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)


# HER
def compare_inference_guiding(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = get_td_from_path(dataset_folder, instance_idx)

    env, td_ignore, generator_params = make_instance(4,4,4,5,50, batch_size=1)

    target_assignments, proc_times, job_id, pos_job = get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")


    inference_assignments = None
    if model_type == "adj":
        # NORMAL INFERENCE
        inference_assignments, elapsed, assignments_over_time = adj_inference(proc_times, job_id, pos_job, adj_model_path, n_samples)
        # INFERENCE GUIDE
        guided_inference_assignments, guided_elapsed, guided_assignments_over_time = guide_adj_inference(proc_times, job_id, pos_job, adj_model_path, n_samples)

    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")


    guided_inference_assignments = inference_assignments[:, :, :4, :16]
    inference_assignments = inference_assignments[:, :, :4, :16]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = show_order_clear(inference_assignments, 16)
    else:
        inference_assignments = round_to_values(inference_assignments, 16)

    print(inference_assignments)
    print(guided_inference_assignments)


    """
    # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :4, :16]
        if order:
            assignment_at_time = round_to_values(assignment_at_time, 16)
        else:
            assignment_at_time = show_order_clear(assignment_at_time, 16)

        if torch.equal(assignment_at_time, inference_assignments):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(inference_assignments)
            break

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    # TODO fjern
    td_scheduled = inferenced_schedule(target_assignments, order, env, td.copy(), graph_save_path)
    # TODO gjør seinere så ikke kommenter ut
    # td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['makespan']
    print(makespan)

    """





model_type = "adj"
order = False
adj_model_path = None
enc_model_path = None
if order:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_2.pth'
else:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_1.pth'

dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'
instance_idx = 10

print("inference result")
get_inference_result(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)
exit()
   






from diffusion import feature_diffusion, adj_diffusion
import sys

model_to_train = None
if len(sys.argv) > 1:
    model_to_train = int(sys.argv[1])
    logging.info(f"Training models nr {model_to_train}")
else:
    logging.info("Please provide a training number!")


model_type = None
order = None
if model_to_train < 3:
    model_type = "adj"
elif model_to_train > 2:
    model_type = "f"
if model_to_train == 2 or model_to_train == 4:
    order = True
elif model_to_train == 1 or model_to_train == 3:
    order = False

print("Using model: type: {type}, order: {order}")
if model_type == None or order == None:
    raise ValueError("That model type dosent exist. pecify one between 1 and 2")

# 1 adj
# 2 adj ordered
# 3 f
# 4 f ordered



from datetime import datetime

def train_models(model_type: str, order: bool):

    jobs = 4
    ma = 4
    ops_per_job = 4
    min_proc = 5
    max_proc = 50

    base_embed = 3
    embed_size = 80

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
    # TODO full path

    lrs = [1e-3, 1e-4, 1e-5, 1e-6]
    testing_epochs = 2
    run_epochs = 100

    training_func = None
    graph_name = None
    if  model_type == "f": 
        training_func = feature_diffusion
    elif model_type == "adj": 
        training_func = adj_diffusion
    graph_name = f"model {model_type}, with order: {order}"

    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'

    loss_image_path = f'{full_path}/models/feature_v_adj'
    graph_save_folder = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs" 

    train_dataset_path = f'{full_path}/data/with_targets/batched_444'
    test_dataset_path = f'{full_path}/data/with_targets/test_batched_444'

    train_dataset = Dataset_RL4CO(train_dataset_path, generator_params, order)
    test_dataset = Dataset_RL4CO(test_dataset_path, generator_params, order)

    model_path_enc = f'{full_path}/models/feature_v_adj/enc_type_{model_type}_order_{order}.pth'
    model_path_adj = f'{full_path}/models/feature_v_adj/adj_type_{model_type}_order_{order}.pth'
    
    best_loss = 1
    best_lr = None

    print(f"Began training model {model_type} order_{order} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    for lr in lrs:
        logging.info(f"Training with lr {lr}")
        path_enc, path_adj, last_epoch_loss = training_func(
            loss_image_path, train_dataset, test_dataset, model_to_train,
            base_embed, embed_size, model_path_enc, model_path_adj,
            graph_name, graph_save_folder, testing_epochs, lr
        ) 
        if best_loss > last_epoch_loss: 
            best_lr = lr
            best_loss = last_epoch_loss

    print(f"Best lr found: {best_lr}, for model {model_type} order_{order} training full model now")
    path_enc, path_adj, last_epoch_loss = training_func(
        loss_image_path, train_dataset, test_dataset, model_to_train,
        base_embed, embed_size, model_path_enc, model_path_adj,
        graph_name, graph_save_folder, run_epochs, best_lr
    )



    print(f"Ended training model {model_type} order_{order} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


train_models(model_type, order)




