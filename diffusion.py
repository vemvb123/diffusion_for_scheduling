import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import random
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data
from torchtyping import TensorType
from torchvision.transforms import Lambda
import torchvision.transforms.functional as F
import PIL
from torchvision.transforms import ToPILImage


from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
from deepinv.models.diffunet import DiffUNet
import os

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import deepinv
import torch.nn as nn
from typing import Tuple
import utils
# import new_dataset

import sys

from diffusers import UNet1DModel




'''
dataset has: image of machine relations (1 for each machinbe)
model outputs: set of coordinates, to one of the coordinates for an operation, where each coordinate in sequence gives the assignment order
target: ideal coordinates
'''
def adj_diffusion(op_n, model_path: str, batch_size: int = 32, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):
    # TODO finn ut av mengde parametere i en modell med ett lag, og hvordan bruke flere lag

    # TODO: Import dataset
    # TODO: Print size of dataset

    model_img = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
    optimizer = torch.optim.Adam(model_img.parameters(), lr=lr)

    model_coords = UNet1DModel(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model_coords.parameters(), lr=lr)

    mse = deepinv.loss.MSE()

    beta_start = lr # antar at skal vere det samme som lr
    # beta_start = 1e-3
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch
        # Instances: the different images, shape (amt images, w, h, b)
        # coords: coordinates, shape (amt cords (3 for x,y,ma), w, b)
        for batch_idx, (instances, coords) in enumerate(train_loader):
            instances = instances.to(device) # TODO: Burde vera i samma storrelse som gjer at man kan sende rett inn mellom kanalane

            # TODO diffuse koordinater
            # Sample random timesteps
            t = torch.randint(0, timesteps, (instances.shape[0],), device=device)

            # Sample noise
            # TODO for 1d diff burde vell det vera ei liste, altsa ikke sann 3 2d mat
            # TODO verifiser at noise ikke bare inneholder 0
            noise = torch.randn_like(torch.zeros(coords))
            # Apply forward diffusion process at timestep t
            noised_coords = (
                sqrt_alphas_cumprod[t, None, None, None] * coords
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )

            # TODO: Send bilda inn i 2d unet, fa tilbake encodete bilder
            optimizer.zero_grad()
            encoded_instances = model(instances, t, type_t='timestep')

            # TODO: flatut enkoda bilder
            # TODO skjekk om data ser riktig ut
            encoded_flat = encoded_instances.flatten(start_dim=1)

            # TODO Concat encoda bilder, og koordinater
            encimg_coords_cat = torch.cat([
                    noised_coords,
                    encoded_flat,
                ], dim=1)

            # TODO send encodete bilder og koordinater inn i modell, fa tilbake koordinater
            # Predict noise
            pred_coords_noise = model_coords(encimg_coords_cat, t, type_t="timestep")
            
            # TODO regn lossm og backpropagate, for 1d og 2d
            loss = nn.MSELoss()(pred_coords_noise, noise)
            loss.backward()
            optimizer.step()


            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            model_path,
        )


    logging.info("saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path

























