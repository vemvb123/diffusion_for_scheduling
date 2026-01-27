import logging
import matplotlib.pyplot as plt
import numpy as np
from deepinv.models.diffunet import DiffUNet

import torch
import os

from torch.utils.data import DataLoader, Subset


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


def mask_invalid(valid_h, valid_w, pred, noise):
    pred_valid  = pred[..., :valid_h, :valid_w]
    noise_valid = noise[..., :valid_h, :valid_w]
    return pred_valid, noise_valid


def get_dataset_loaders(train_dataset, test_dataset, batch_size: int = 32, subset: bool = False):
    train_loader, test_loader = None, None

    if subset:
        train_subset_dataset = Subset(train_dataset, range(batch_size*5))
        test_subset_dataset = Subset(test_dataset, range(batch_size*5))
        train_loader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")
    return train_loader, test_loader


def get_models(model_type: str, n_base_features: int, n_embed_features, lr, device: str = "cuda"):
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
         model_adj = DiffUNet(
            in_channels=n_base_features + 1,
            out_channels=1,
            pretrained=None
        ).to(device)
         optimizer = torch.optim.Adam(model_adj.parameters(), lr=lr)

    return model_adj, model_enc, optimizer