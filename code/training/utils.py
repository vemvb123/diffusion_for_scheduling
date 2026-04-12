import logging
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
from deepinv.models.diffunet import DiffUNet

import torch
import os

from torch.utils.data import DataLoader, Subset, random_split




def save_losses(epoch, graph_name, losses):
    os.makedirs("./losses", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)
    os.makedirs("./img/noised", exist_ok=True)
    os.makedirs("./img/denoised", exist_ok=True)
    os.makedirs("./img/original", exist_ok=True)

    # all_losses.extend(epoch_losses)

    # Save the losses list after each epoch
    np.save(f"./losses/losses_epoch_{epoch+1}_{graph_name}.npy", np.array(losses))


def plot_losses(save_path, graph_name, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(graph_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{save_path}/{graph_name}.png")


def mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj):
    # 1. Crop spatially
    pred_valid  = pred[..., :valid_h, :valid_w]
    noise_valid = noise[..., :valid_h, :valid_w]

    if ops_ma_adj is None:
        return pred_valid, noise_valid

    ops_ma_adj_valid = ops_ma_adj[..., :valid_h, :valid_w]

    # 2. If ops_ma_adj has at least one valid cell, apply it
    if ops_ma_adj_valid.any():
        # Ensure mask is boolean
        mask = ops_ma_adj_valid.bool()

        # Select only valid cells
        pred_valid  = pred_valid[mask]
        noise_valid = noise_valid[mask]

    # 3. If ops_ma_adj is all zeros → ignore it
    return pred_valid, noise_valid


"""

def get_dataset_loaders(train_dataset, test_dataset, batch_size: int = 32, val_ratio: float = None, subset: bool = False):
    train_loader, test_loader = None, None

    if subset:
        train_subset_dataset = Subset(train_dataset, range(batch_size*5))
        test_subset_dataset = Subset(test_dataset, range(batch_size*5))
        train_loader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")

    if val_ratio != None:
        n_total = len(train_dataset)
        n_val   = int(val_ratio * n_total)         # e.g., 20% validation
        n_train = n_total - n_val

        train_subset, val_subset = random_split(train_dataset, [n_train, n_val])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

        logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
        logging.info(f"N instances in val dataset: { len(val_loader.dataset) }")

        return train_loader, test_loader, val_loader


    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    return train_loader, test_loader

"""


import logging
import random
from torch.utils.data import DataLoader, Subset, random_split


def get_dataset_loaders(
    train_dataset,
    test_dataset,
    batch_size: int = 32,
    val_ratio: float = None,
    subset: bool = False
):
    train_loader, test_loader = None, None

    # Only reduce dataset IF subset=True
    if subset:
        subset_ratio = 0.05  # 20%

        train_size = int(len(train_dataset) * subset_ratio)
        test_size = int(len(test_dataset) * subset_ratio)

        train_indices = random.sample(range(len(train_dataset)), train_size)
        test_indices = random.sample(range(len(test_dataset)), test_size)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Normal loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"N instances in test dataset: {len(test_loader.dataset)}")

    # Validation split
    if val_ratio is not None:
        n_total = len(train_dataset)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_val

        train_subset, val_subset = random_split(train_dataset, [n_train, n_val])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        logging.info(f"N instances in train dataset: {len(train_loader.dataset)}")
        logging.info(f"N instances in val dataset: {len(val_loader.dataset)}")

        return train_loader, test_loader, val_loader

    logging.info(f"N instances in train dataset: {len(train_loader.dataset)}")

    return train_loader, test_loader




def get_models(model_type: str, n_base_features: int, n_embed_features, lr, device: str = "cuda", path = None):
    model_adj, model_enc, optimizer = None, None, None

    if model_type == "f":
        model_enc = DiffUNet(
            in_channels=n_base_features,
            out_channels=n_embed_features,
            pretrained=None
        ).to(device)

        model_adj = DiffUNet(
            in_channels=n_embed_features+ 1,
            out_channels=1,
            pretrained=None
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model_enc.parameters()) + list(model_adj.parameters()),
            lr=lr
        )

    elif model_type == "adj":
         # im loading this
         model_adj = DiffUNet(
            in_channels=n_base_features + 1,
            out_channels=1,
            pretrained=None
        ).to(device)
         optimizer = torch.optim.Adam(model_adj.parameters(), lr=lr)


    if path != None:
        # im loading this
        if os.path.exists(path):
            '''
            checkpoint = torch.load(path, map_location=device)
    
            model_adj.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            '''

            model_adj = deepinv.models.DiffUNet(
                in_channels=n_base_features + 1, out_channels=1, pretrained=Path(path)
            ).to(device)
            optimizer = None

    return model_adj, model_enc, optimizer


def count_nan_indices(loader):
    total_nans = 0

    for batch in loader:
        target_assignments, proc_times, job_ops_adj, ops_ma_adj = batch

        B = target_assignments.shape[0]

        for i in range(B):
            if (
                torch.isnan(target_assignments[i]).any() or
                torch.isnan(proc_times[i]).any() or
                torch.isnan(job_ops_adj[i]).any() or
                torch.isnan(ops_ma_adj[i]).any()
            ):
                total_nans += 1
    logging.info(f"Total instances with NaN values: {total_nans}")
    return total_nans