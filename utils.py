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

# from rl4co.models.zoo.l2d.policy import L2DPolicy
# from rl4co.models.zoo.l2d.decoder import L2DDecoder
# from rl4co.models.nn.graph.hgnn import HetGNNEncoder
# from rl4co.utils.trainer import RL4COTrainer
# from IPython.display import display, clear_output

# logging.info(2)
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


import gc
from rl4co.envs import JSSPEnv
from rl4co.models.zoo.l2d.model import L2DPPOModel
from rl4co.models.zoo.l2d.policy import L2DPolicy4PPO
from torch.utils.data import DataLoader
import json
import os


# 1: no order
# 2: ordered
# 3: ordered ma

def schedule_actions(
        actions, td_unscheduled, env, ordered: int
):
    """
    ordered == 2: global normalized order (i / n_actions)
    ordered == 3: per-row normalized order
    ordered == 1: (reserved / undefined – keep same as 1 for now)
    """

    if ordered not in (1, 2, 3):
        raise ValueError(f"ordered {ordered} is not supported, must be either 1, 2 or 3")

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

            if ordered == 1 or ordered == 2:
                # global normalized order
                normalized_order = i / float(n_actions)
                assignment_adj[diff] = normalized_order

            elif ordered == 3:
                # for each row, assign per-row sequential rank
                batch_idx, rows, cols = diff.nonzero(as_tuple=True)
                if batch_idx.numel() > 0 and batch_idx.max() > 0:
                        # optionally handle multi-batch later
                        raise NotImplementedError("Batch size > 1 not yet supported for ordered==3")


                for r, c in zip(rows.tolist(), cols.tolist()):
                    # increment this row's counter
                    per_row_counts[r] += 1

                    # total possible edges for that row
                    # (could also compute it if known in advance)
                    # But here we only know relative rank, not full normalizer
                    # So store raw sequence index for now
                    assignment_adj[r, c] = per_row_counts[r]

        prev_adj = new_adj.clone()

    if ordered == 3:
        # now normalize per row
        # for each row r, divide all nonzero entries
        # by the maximum count
        for r in range(assignment_adj.size(0)):
            row_vals = assignment_adj[r]
            nonzero = row_vals.nonzero()
            if nonzero.numel() > 0:
                max_val = row_vals.max()
                if max_val > 0:
                    assignment_adj[r] = assignment_adj[r] / float(max_val)

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
def get_feature_adj_from_instance(td: TensorDict, env, ordered: int) -> tuple[
        torch.Tensor, # target assignments
        torch.Tensor, # proc times matrix
        torch.Tensor, # jobid matrix
        torch.Tensor # pos in job matrix
        ]:
    # bytt senere ut med assignments fra target
    assignments = None
    if ordered == 2 or ordered == 3:
        td_scheduled = schedule_actions(td['opt_actions'], td.copy(), env, ordered)
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
    if torch.isnan(assignments).any():
        print("assignemtn NaN values found in features tensor")
        exit()
    if torch.isnan(proc_times).any():
        print("proc NaN values found in features tensor")
        exit()
    if torch.isnan(job_id).any():
        print("job id NaN values found in features tensor")
        exit()
    if torch.isnan(pos_job).any():
        print("pos job NaN values found in features tensor")
        exit()





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

import bisect

class Dataset_RL4CO(Dataset):
    def __init__(self, folder, ordered: bool, generator_params, transform=None):
        self.folder = folder
        self.transform = transform
        self.ordered = ordered
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
                td_instance, self.env, self.ordered
            )

        return target_assignments, proc_times, job_id, pos_job



from inference import feature_inference, adj_inference

def round_to_values(x, ordered: bool, n_values: int):
    if ordered == False:
        # flatten tensor, get indices of top 16 values
        topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

        # create a copy or a zero tensor
        y = x.clone()

        # set the top 16 values to 1
        y.flatten()[topk_idx] = 1.0
        return y



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



def map_assignemnts_to_actions(assignments, ordered: bool):
    if ordered:

        # remove batch/channel dims if present
        if assignments.dim() == 4:
            assignments = assignments.squeeze(0).squeeze(0)  # (4,16)

        H, W = assignments.shape  # H=4, W=16
        section_width = 4
        num_sections = W // section_width

        actions = []

        if ordered:
            # order by largest value first
            _, indices = torch.topk(assignments.flatten(), H * W)

            for idx in indices:
                row = idx // W
                col = idx % W
                section = col // section_width
                action = section * H + row + 1
                actions.append(action.item())

    else:
        # section-wise column sweep:
        # col 0 in section 0, col 0 in section 1, ...
        # then col 1 in section 0, etc.
        for local_col in range(section_width):
            for section in range(num_sections):
                col = section * section_width + local_col

                # pick row with max value in this column
                row = torch.argmax(assignments[:, col]).item()

                action = section * H + row + 1
                actions.append(action)

    return actions



def make_step(env, td, action):
    td['action'] = torch.tensor([action])
    td = env.step(td)['next']
    return td

def inferenced_schedule(assignments, ordered: bool, env, td, path_save_image: str):
    actions = map_assignemnts_to_actions(assignments, ordered)

    if path_save_image:
        env.render(td, 0)
        i = 0
        fig = None
        while not td["done"].all():
            td = make_step(env, td, actions[i])
            fig = env.render(td, 0)
            i += 1
        fig.savefig(f"frame_{i:03d}.png")
        plt.close(fig)
    else:
        i = 0
        while not td["done"].all():
            td = make_step(env, td, actions[i])
            i+=1

    return td



 
# /cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444/0_184.pt
def get_td_from_path(path, instance_idx: int) -> TensorDict:
    for fname in os.listdir(path):
        if not fname.endswith(".pt"):
            continue

        start, end = map(int, fname.replace(".pt", "").split("_"))

        if start <= instance_idx < end:
            file_path = os.path.join(path, fname)
            batch = torch.load(file_path)

            local_idx = instance_idx - start
            return batch[local_idx]

    raise ValueError(f"Instance {instance_idx} not found in {path}")


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





def get_inference_result():

    path = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444"
    which_instance_in_batch = 10

    td = get_td_from_path(path, which_instance_in_batch)
    env, td_ignore, generator_params = make_instance(4,4,4,5,50, batch_size=1)
    ordered = 1

    target_assignments, proc_times, job_id, pos_job = \
        get_feature_adj_from_instance(
            td, env, ordered
        )


    embed_size = 80
    n_samples = 1
    # TODO må endre disse stiene, dette er bare fyllekode
    model_path_enc_ordered = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/enc_type_2.pth'
    model_path_adj = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_2.pth'

    # TODO tar først modell som har order
    # f_ordered_assignments, elapsed = feature_inference(proc_times, job_id, pos_job, model_path, n_samples, embed_size)    
    # f_assignments, elapsed = feature_inference(proc_times, job_id, pos_job, model_path, n_samples, embed_size)    
    adj_ordered_assignments, elapsed = adj_inference(proc_times, job_id, pos_job, model_path_adj, n_samples, embed_size)    
    # adj_assignments, elapsed = adj_inference(proc_times, job_id, pos_job, model_path, n_samples, embed_size)    
    adj_ordered_assignments = adj_ordered_assignments[:, :, :4, :16]
    adj_assignments = adj_assignments[:, :, :4, :16]

    # adj_assignment = round_to_values(adj_assignments, False, 16)
    # trenger ikke runde ordered, bare bruker til 16 største når skedulerer
    # TODO ma assignment er ikke assignet, må muligens fjerne opt actions og optma assignments fra td
    td_scheduled = inferenced_schedule(adj_assignments, False, env, td.copy())

    makespan = td_scheduled['makespan']

    # sammenligner target og fra modell
    # her er det ikke viktig at resultatene er like, siden taget gir ikek en perfekt løsning
    get_clear_sequence(td['opt_ma_assignments'])
    get_clear_sequence(adj_assignments)


    






from diffusion import feature_diffusion, adj_diffusion
import sys

model_to_train = None
if len(sys.argv) > 1:
    model_to_train = int(sys.argv[1])
    logging.info(f"Training models nr {model_to_train}")
else:
    logging.info("Please provide a training number!")

# 1 features
# 2 features ordered
## 3 features ordered ma
# 4 adj
# 5 adj ordered
## 6 adj ordered ma

from datetime import datetime

def train_models(model_to_train: int):
    logging.info(f"Training models nr {model_to_train}")

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
    if model_to_train >= 3: 
        training_func = feature_diffusion
        graph_name = f"feature vector model {model_to_train}"
    else: 
        training_func = adj_diffusion
        graph_name = f"adjecency model {model_to_train}"

    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'

    loss_image_path = f'{full_path}/models/feature_v_adj'
    graph_save_folder = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs" 

    train_dataset_path = f'{full_path}/data/with_targets/batched_444'
    test_dataset_path = f'{full_path}/data/with_targets/test_batched_444'
    train_dataset = Dataset_RL4CO(train_dataset_path, model_to_train, generator_params)
    test_dataset = Dataset_RL4CO(test_dataset_path, model_to_train, generator_params)

    model_path_enc = f'{full_path}/models/feature_v_adj/enc_type_{model_to_train}.pth'
    model_path_adj = f'{full_path}/models/feature_v_adj/adj_type_{model_to_train}.pth'
    
    best_loss = 1
    best_lr = None

    logging.info(f"Began training model {model_to_train} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

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

    logging.info(f"Best lr found: {best_lr}, for model {model_to_train} training full model now")
    path_enc, path_adj, last_epoch_loss = training_func(
        loss_image_path, train_dataset, test_dataset, model_to_train,
        base_embed, embed_size, model_path_enc, model_path_adj,
        graph_name, graph_save_folder, run_epochs, best_lr
    )



    logging.info(f"Ended training model {model_to_train} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")


train_models(model_to_train)




