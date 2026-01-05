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
from utils import ImageCoordinateDataset, encode_image

def diffusion(op_n, model_path: str, batch_size: int = 32, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):

    dataset = ImageCoordinateDataset("tmp_dataset/img_coords_dataset")
    loader = DataLoader(dataset, batch_size=5, shuffle=True)


    logging.info(f"Instances in dataset: { len(loader.dataset) }")
    logging.info(f"Instances in loader: { len(loader) }")
    
    # TODO endre bilde rep, til ett eneste bilde, et bilde som kan encode hele problemet
    # out channels, flere, samme som embedding dimensjon .. men burde kanskje endre bilderep, fordi nu sliter koordinatet
    model_img = deepinv.models.DiffUNet(in_channels=1, out_channels=16, pretrained=None).to(device)
    model_coords = UNet1DModel(in_channels=4, out_channels=3).to(device)
    
    # printing amount of parameters
    trainable_params = sum(p.numel() for p in model_img.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    optimizer_img = torch.optim.Adam(model_img.parameters(), lr=lr)
    optimizer_coords = torch.optim.Adam(model_coords.parameters(), lr=lr)

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
        # (imgs1, imgs2, imgs3, imgs4), (col1, col2, col3) = batch
        for batch_idx, (imgs, coords) in enumerate(loader):
            # TODO tester shapes, sa bruker bare fyrste bilde
            logging.info('shait')
            imgs_cat = imgs[0]
            logging.info(imgs_cat.shape)
            # imgs_cat = torch.cat(imgs, dim=1)  # shape: (B, 4, H, W)
            # logging.info(imgs_cat)
            coords_cat = torch.cat(coords, dim=1)  # shape: (B, 4, H, W)

            # Concatinating for input into channels
            imgs_cat = imgs_cat.to(device, dtype=torch.float32)
            coords_cat = coords_cat.to(device, dtype=torch.float32)
            coords_cat = coords_cat.squeeze(2)

            # TODO : Pass pa at alt av verdier er normaliser
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device) 
            logging.info('t shape')
            logging.info(t.shape)

            # encode images
            optimizer_img.zero_grad()
            enc_imgs = model_img(imgs_cat, t, type_t="timestep")
            logging.info("kjort modell")
            logging.info(enc_imgs.shape)

            # Sample noise
            noise = torch.randn_like(coords_cat)
            # Apply forward diffusion process at timestep t
            noised_coords = (
                sqrt_alphas_cumprod[t, None, None, None] * coords_cat
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )


            # TODO: flatut enkoda bilder
            # TODO skjekk om data ser riktig ut
            encoded_flat = enc_imgs.flatten(start_dim=2)

            # TODO Concat encoda bilder, og koordinater
            # TODO skjekk at shape blir riktig
            logging.info("catination")
            logging.info(noise.shape)
            logging.info(encoded_flat.shape)
            return
            encimg_coords_cat = torch.cat([
                    noise,
                    encoded_flat,
                ], dim=1)
            
            logging.info(encimg_coords_cat.shape)
            return 

            # TODO send encodete bilder og koordinater inn i modell, fa tilbake koordinater
            # Predict noise
            pred_coords_noise = model_coords(encimg_coords_cat, t, type_t="timestep")
            
            # TODO regn lossm og backpropagate, for 1d og 2d .. hvordan ta loss fra en modell, og gi til den andre..
            loss = nn.MSELoss()(pred_coords_noise, noise)
            loss.backward()
            optimizer_coords.step()
            optimizer_img.step()

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








print("running")
diffusion(1, "..", 5, 2)
















