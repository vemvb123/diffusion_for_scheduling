import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time
import networkx as nx
import matplotlib.pyplot as plt
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.models.zoo.l2d.decoder import L2DDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.utils.trainer import RL4COTrainer

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import scheduling_utils
# logging.info(1)
import diffusion
import inference
import importlib
# logging.info(2)
importlib.reload(diffusion)
importlib.reload(inference)
importlib.reload(scheduling_utils)
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



def main():
    logging.info("starting program")
    # bruk AdjDataset til å lage instansene
    instance_size = (4,4,4)
    pad_size=(20,20)
    root_dir = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/ad_test/"
    dataset = scheduling_utils.AdjDataset(root_dir, instance_size, pad_size)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    # referer til en enkeltinstanse, der du da får adj, procs, h, w
    adj, proc_instance, w, h = dataset[7]
    logging.info(proc_instance)
    # putt procs inn i inference
    model_path = "models/adj_444_0.0001.pth"
    n_samples = 1
    dim_x = 16
    dim_y=16
    pad_x=20
    pad_y=20
    x, time = inference.adj_inference(proc_instance, model_path, n_samples, dim_x, dim_y, pad_x, pad_y)
    # rund opp de n største verdiene
    logging.info("fasan i satan")
    x = x[0, 0, :16, :16]
    x = x.clone()
    vals, idxs = torch.topk(x.view(-1), k=12)
    out = torch.zeros_like(x)
    out.view(-1)[idxs] = 1.0
    logging.info(out)
    # sammenlign produsert x, med adj
    adj = adj[:dim_x, :dim_y] 
    logging.info(adj)
    # utils.make_dataset_adjacency()
    """
    instance_size = (4,4,4)
    pad_size = (20,20)
    lrs = [1e-6]
    num_epochs=50
    for lr in lrs:
        model_path_to_make=f"models/adj_{instance_size[0]}{instance_size[1]}{instance_size[2]}_{lr}.pth"
        logging.info(f"training model {model_path_to_make}")
        diffusion.adj_diffusion(instance_size, pad_size, model_path_to_make, lr=lr, num_epochs=num_epochs)

    """
    logging.info("program finished")


if __name__ == "__main__":
    main()




# utils.make_dataset_adjacency()


# utils.make_schedule_from_assignment_444_fasan("", "")
# utils.make_schedule_from_assignment("", "")

"""
is_ready
start_op_per_job
num_eligible
next_op
ma_assignment
lbs

td :
TensorDict(
    fields={
        action_mask: Tensor(shape=torch.Size([1, 26]), device=cpu, dtype=torch.bool, is_shared=False),
        busy_until: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        end_op_per_job: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.int64, is_shared=False),
        finish_times: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        is_ready: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.bool, is_shared=False),
        job_done: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.bool, is_shared=False),
        job_in_process: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.bool, is_shared=False),
        job_ops_adj: Tensor(shape=torch.Size([1, 5, 10]), device=cpu, dtype=torch.int64, is_shared=False),
        lbs: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        ma_assignment: Tensor(shape=torch.Size([1, 5, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        next_op: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.int64, is_shared=False),
        num_eligible: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        op_scheduled: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.bool, is_shared=False),
        ops_adj: Tensor(shape=torch.Size([1, 10, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        ops_job_map: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.int64, is_shared=False),
        ops_ma_adj: Tensor(shape=torch.Size([1, 5, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        ops_sequence_order: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.int64, is_shared=False),
        pad_mask: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.bool, is_shared=False),
        proc_times: Tensor(shape=torch.Size([1, 5, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        start_op_per_job: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.int64, is_shared=False),
        start_times: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        time: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)

"""


"""
    logging.info('running')


    class GrayscaleImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.image_paths = sorted([p for p in self.root_dir.rglob('*')
                                       if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('L')  # convert to grayscale 'L' mode
            if self.transform:
                img = self.transform(img)
            return img

# Define transforms: convert to tensor and optionally normalize
    transform = transforms.Compose([
        transforms.ToTensor(),   # result will be shape [1, H, W] since grayscale
        # you can add normalization if needed
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Create dataset
    dataset = GrayscaleImageDataset(
        root_dir="/home/vemund/Dokumenter/koding/datasets/plugAndPlay/images/",
        transform=transform
    )

# Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

# Example: show one batch
    for batch in train_loader:
        logging.info(batch.shape)  # should logging.info something like [32, 1, H, W]
        break


    logging.info('donald')


exit()

"""





"""
logging.info('running')
utils.check_gpu()

utils.make_dataset()

logging.info("exiting program")
"""

# mean = (0.5, 0.5, 0.5)
#std  = (0.5, 0.5, 0.5)
# image_size=64

"""
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2])),
])
"""
"""
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),  # For grayscale; change if RGB
])




batch_size = 32
num_epochs = 100
path = 'instances/'

logging.info("starting here")

lr_to_try = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
for lr in lr_to_try:
    model_path = f'diffusionAsPlugAndPlay_lr_masks_{lr}.pth'
    dataset = utils.CustomImageDataset(path, "cuda", transform, 25000)
    train_loader = DataLoader(dataset, batch_size, True)
    logging.info(f'training model {model_path}')
    diffusion.apply_diffusion(batch_size, train_loader, num_epochs, model_path, lr)

logging.info("done training. Exiting program")
exit()

# logging.info(train_loader.dataset[0].shape)
# img = train_loader.dataset[0]  # shape [1, 16, 16]
# img2 = img.squeeze(0)          # shape [16, 16]

# plt.imshow(img2.cpu().numpy(), cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.show()
    # diffusion.apply_diffusion(batch_size, train_loader, num_epochs, model_path, lr)



"""
"""
betas_start = [0.1, 0.01, 0.001, 0.0001]
for beta_start in betas_start:
    model_path = f'diffusionAsPlugAndPlay_lr_{beta_start}.pth'
    x = inference.apply_inference(16, 16, model_path, 1, beta_start)

"""
"""
logging.info('done training all models')
exit()

generator_params = {
"num_jobs": 4,  # the total number of jobs
"num_machines": 4,  # the total number of machines that can process operations
"min_ops_per_job": 4,  # minimum number of operatios per job
"max_ops_per_job": 4,  # maximum number of operations per job
"min_processing_time": 5,  # the minimum time required for a machine to process an operation
"max_processing_time": 20,  # the maximum time required for a machine to process an operation
"min_eligible_ma_per_op": 1,  # the minimum number of machines capable to process an operation
"max_eligible_ma_per_op": 2,  # the maximum number of machines capable to process an operation
}

batch_size = 300
embed_dim = 32
encoder_num_layers = 4
epochs = 100

# 1, 20, 16

# Encoding operations and machines

# num of wanted instances

env = FJSPEnv(
    generator_params=generator_params, 
    _torchrl_mode=True, 
    stepwise_reward=True
)
# env = FJSPEnv(generator_params=generator_params)


td = env.reset(batch_size=[batch_size])

logging.info(td)
exit()



encoder = HetGNNEncoder(embed_dim=embed_dim, num_layers=encoder_num_layers)
(ma_emb, op_emb), init = encoder(td)
# logging.info("Shapes of original embeddings")
# logging.info(f"Ma: {ma_emb.shape}")
# logging.info(f"Op: {op_emb.shape}")

# Combining embeddings
dataset = utils.EmbeddingDataset(op_emb, ma_emb, td, env)
logging.info(f"Shape of a combined embedding instance: {dataset[0].shape}")
# Diffusion process
model_path = 'trained_diffusion_model_makespan.pth'
diffusion.apply_diffusion_makespan(batch_size, dataset, epochs, model_path)

# model_path = 'trained_diffusion_model.pth'
diffusion.apply_diffusion(batch_size, dataset, epochs, model_path)

# Applying inference
dim_x = op_emb.shape[1] + ma_emb.shape[1]
dim_y = embed_dim
n_samples = 5
# samples = inference.apply_inference(dim_x, dim_y, model_path, n_samples)
# logging.info(f"Shapes of an inferenced sample instance: {samples.shape}")

# Decompositioning op and ma embeddings
# s_op_emb, s_ma_emb = dataset.decomposition_instance(samples)
# logging.info(f"Decompositioned op embedding: {s_op_emb.shape}")
# logging.info(f"Decompositioned ma embedding: {s_ma_emb.shape}")


# Visulizing schedule
# utils.visulize_schedule(td, env, embed_dim, s_ma_emb, s_op_emb)


def make_samples():
    # Applying inference
    dim_x = op_emb.shape[1] + ma_emb.shape[1]
    dim_y = embed_dim
    n_samples = 32
    samples = inference.apply_inference(dim_x, dim_y, model_path, n_samples)
    logging.info(f"Shapes of an inferenced sample instance: {samples.shape}")
    return samples

def makespan_of_generated_schedules(samples):
    # Decompositioning op and ma embeddings
    s_op_emb, s_ma_emb = dataset.decomposition_instance(samples)
    logging.info(f"Decompositioned op embedding: {s_op_emb.shape}")
    logging.info(f"Decompositioned ma embedding: {s_ma_emb.shape}")
        
    # Visulizing schedule
    makespan_batches = utils.visulize_schedule(td, env, embed_dim, s_ma_emb, s_op_emb)
    return makespan_batches

def avg_makespan_of_bacthes(makespans):
    return makespans.mean()
"""

"""
samples = make_samples()
makespan_batches = makespan_of_generated_schedules(samples)

logging.info(makespan_batches)
avg_makespan = avg_makespan_of_bacthes(makespan_batches)
logging.info(avg_makespan)
"""


