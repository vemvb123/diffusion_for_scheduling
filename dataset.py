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


import bisect

from tensordict import TensorDict, from_dict
from typing import Callable, Dict, List, Tuple


import torch.nn.functional as F


from scheduling_utils import make_adj_with_order










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




def make_dataset(n: int, dataset_folder: str, 
                 n_jobs, n_ma, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, 
                 target_model: str, batch_size: int = 184):

    logging.info("Making dataset...")

    os.makedirs(dataset_folder, exist_ok=True)
    for i in range(0, n, batch_size):
        # lag instanse

        env, td, generator_params = make_instance(n_ma=n_ma, n_jobs=n_jobs, 
                                                  max_op_per_job=max_op_per_job, min_op_per_job=min_op_per_job, 
                                                  max_proc_time=max_proc_time, min_proc_time=min_proc_time, 
                                                  max_eligable_ma_per_op=n_ma, min_eligable_ma_per_op=n_ma, 
                                                  batch_size=batch_size)
        # fa target fra instance 
        td_target, actions = make_target(env, td.copy(), True, target_model)
        # lagre json med: td, og optimale td koords
        td.set('opt_assignment', td_target['ma_assignment'])
        td.set('opt_actions', torch.tensor(actions))
        # lagre coords i en json, med visse navn
        torch.save(td.copy(), f'{dataset_folder}/{i}_{i+batch_size}.pt')

        print(f'Made instance {i} to {i+batch_size}')

    print(f'Made all {i+batch_size} instances. Done making dataset')









def get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance: str) -> Dict:

    with open(filepath_brandimarte_instance, "r") as f:
        lines = [line.strip() for line in f if line.strip()]


    # First line: number of jobs, number of machines
    first = lines[0].split()
    n_jobs = int(first[0])
    n_machines = int(first[1])

    # Stats
    global_min_pt = float("inf")
    global_max_pt = float("-inf")
    min_ops = float("inf")
    max_ops = float("-inf")

    # New stats for machine options per operation
    min_machine_options = float("inf")
    max_machine_options = float("-inf")

    # Loop through job lines
    for i in range(1, 1 + n_jobs):
        parts = list(map(int, lines[i].split()))
        idx = 0

        # Number of operations in this job
        n_ops = parts[idx]
        idx += 1

        # Update min/max number of operations
        min_ops = min(min_ops, n_ops)
        max_ops = max(max_ops, n_ops)

        # Loop through each operation
        for _ in range(n_ops):
            m_count = parts[idx]
            idx += 1

            # Track machine options stats
            min_machine_options = min(min_machine_options, m_count)
            max_machine_options = max(max_machine_options, m_count)

            # m_count pairs of (machine, processing time)
            for _ in range(m_count):
                machine_id = parts[idx]        # machine index (not needed for stats)
                proc_time = parts[idx + 1]     # processing time
                idx += 2

                # Track processing time
                global_min_pt = min(global_min_pt, proc_time)
                global_max_pt = max(global_max_pt, proc_time)


    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "min_processing_time": global_min_pt,
        "max_processing_time": global_max_pt,
        "fewest_operations": min_ops,
        "most_operations": max_ops,
        "min_machine_options": min_machine_options,
        "max_machine_options": max_machine_options
    }





def main():
    pass
    ### Lag dataset
    """
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'

    test_size = 20000
    train_size = 100000
    n = test_size

    batch_size = 184

    make_dataset(20000)
    """


    ### Dekod parameterverdier for Brandimarte instanse
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
    parameters = get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)
    return 

    test_size = 20000
    train_size = 100000
    n = train_size + test_size

    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/batched_mk01_10j_6ma_6op'
    make_dataset(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
    )



main()


# train_models(model_type, order)





