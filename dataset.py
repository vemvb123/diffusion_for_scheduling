"""
dataset.py contains code for making a dataset of instances
"""



import logging
import os
import torch
from torch.utils.data import Dataset
from rl4co.envs import FJSPEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


from scheduling_utils import make_instance, make_target

from datetime import datetime

import bisect

from tensordict import TensorDict, from_dict
from typing import Callable, Dict, List, Tuple

from diffusion import diffusion, diffusion

import torch.nn.functional as F

import sys




# 1 adj
# 2 adj ordered
# 3 f
# 4 f ordered

# maps file execution parameter (1-4) to some model to train
model_to_train = None
if len(sys.argv) > 1:
    model_to_train = int(sys.argv[1])
    logging.info(f"Training models nr {model_to_train}")
else:
    logging.info("Please provide a training number!")


def map_file_parameter_to_model_type(model_to_train: int) -> Tuple[str, bool]:
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

    return model_type, order

map_file_parameter_to_model_type(model_to_train)










def expand_matrix(x: torch.Tensor, shape_to_make: Tuple[int, int], max_and_min: Tuple[int, int]) -> torch.Tensor:
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



def get_feature_adj_from_instance(td: TensorDict, env, order: bool) -> Tuple[
        torch.Tensor, # target assignments
        torch.Tensor, # proc times matrix
        torch.Tensor, # jobid matrix
        torch.Tensor # pos in job matrix
        ]:

    assignments = None
    if order:
        td_scheduled = make_adj_with_order(td['opt_actions'], td.copy(), env, order)
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

    return assignments, proc_times, job_id, pos_job






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




def make_dataset(n: int, dataset_folder: str, batch_size: int):
    logging.info("Making dataset...")

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
        training_func = diffusion
    elif model_type == "adj": 
        training_func = diffusion
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




dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'

test_size = 20000
train_size = 100000
n = test_size

batch_size = 184

make_dataset(20000)


# train_models(model_type, order)





