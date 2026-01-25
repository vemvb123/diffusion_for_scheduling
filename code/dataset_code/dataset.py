import code.dataset_code.utils as utils

import torch
from rl4co.envs import FJSPEnv
from torch.utils.data import Dataset


import bisect
import os


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
        h = 20
        w = 60
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
            utils.get_feature_adj_from_instance(
                td_instance, self.env, self.order
            )

        for data in [target_assignments, proc_times, job_id, pos_job]:
            if torch.isnan(data).any():
                raise ValueError("Assignment NaN values found in tensor")

        return target_assignments, proc_times, job_id, pos_job