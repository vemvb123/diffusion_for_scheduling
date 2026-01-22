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



