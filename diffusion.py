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




def mean_weight_value(model):
    total = 0.0
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.data.mean()
            count += 1
    return total / count if count > 0 else float('nan')



# GJOR MASKING SLIK:
# LEGGER INN EN MASKE, DER ALT SOM SKAL MASKES HAR VERDI 1
# LAG EN MASKE
# REGN UT LOSS MAP
# FRA LOSS MAPPET, NULL UT DE STEDENE DER MASKEN ER, SLIK AT MASKEREGIONENE IKKE BIDRAR I LOSS
# SÅ TA LOSS MEAN
def adj_diffusion(instance_size, pad_size, model_path: str, batch_size: int = 32, data_dim_x_y: int = 16, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):

    # VURD
    mask_value = 1
    # VURD
    root_dir = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/instances/adj/'

    dataset = utils.AdjDataset(root_dir, instance_size, pad_size)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # x, x2 = dataset[10]

    model = deepinv.models.DiffUNet(in_channels=2, out_channels=1, pretrained=None).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    

    prev_loss = 0
    all_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch
        
        loop = tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False)

        for batch_idx, (adj, procs, adj_unpad_shape, procs_unpad_shape) in enumerate(loop):


            orig_h = adj_unpad_shape[0][0] # 8x8, oppgitt med en liste for hvert elem, en en loste med 8, sa enda en liste med 8
            orig_w = adj_unpad_shape[1][0] # 8
            #       logging.info("\n" + f"sizes: {orig_h}, {orig_w}")
            x_procs_undpad_shape = procs_unpad_shape[1][0] # 4
            y_procs_undpad_shape = procs_unpad_shape[1][0] # 4

            adj = adj.to(device)
            procs = procs.to(device)
            adj = adj.unsqueeze(1)
            procs = procs.unsqueeze(1)

            to_check = torch.cat([adj, procs], dim=1)
            if torch.isnan(to_check).any() or torch.isinf(to_check).any():
                # logging.info("adj or procs contains invalid values!")

                bad_mask = torch.isnan(to_check) | torch.isinf(to_check)
                bad_positions = bad_mask.nonzero()
                bad_batch_indices = bad_positions[:, 0].unique().tolist()

                # print("Bad batch sample indices:", bad_batch_indices)

                for bad_idx in bad_batch_indices:
                    # try random replacements until one is valid
                    while True:
                        random_idx = random.randrange(len(dataset))
                        new_adj, new_procs, new_adj_shape, new_proc_shape = dataset[random_idx]

                        # move to device + add batch dim
                        new_adj = new_adj.to(device).unsqueeze(0)
                        new_procs = new_procs.to(device).unsqueeze(0)

                        # check validity
                        candidate = torch.cat([new_adj, new_procs], dim=1)
                        if not torch.isnan(candidate).any() and not torch.isinf(candidate).any():
                            break

                    adj[bad_idx]   = new_adj
                    procs[bad_idx] = new_procs



            # Sample random timesteps
            t = torch.randint(0, timesteps, (adj.shape[0],), device=device)

            # Sample noise
            noise = torch.randn_like(adj)






            # Apply forward diffusion process at timestep t
            noised_adj = (
                sqrt_alphas_cumprod[t, None, None, None] * adj
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )
            # CONDITION
            model_input = torch.cat([
                noised_adj,
                procs,
            ], dim=1)






            # LOSS
            optimizer.zero_grad()
            noise_pred = model(model_input, t, type_t="timestep")
            loss_map = (noise_pred - noise) ** 2  # shape = (32,1,16,16)
            # loss = mse(noise_pred, noise)
            
            if torch.isnan(noise_pred).any():
                logging.info("NaN in noise_pred!")
            if torch.isinf(noise_pred).any():
                logging.info("Inf in noise_pred!")
            if torch.isnan(noise).any() or torch.isinf(noise).any():
                logging.info("Noise contains invalid values!")
            if torch.isnan(model_input).any() or torch.isinf(model_input).any():
                logging.info("model input contains invalid values!")
                # boolean mask of invalid values
                bad_mask = torch.isnan(model_input) | torch.isinf(model_input)
                # find indices where bad_mask is True
                bad_positions = bad_mask.nonzero()
                # the first column is the batch index
                bad_batch_indices = bad_positions[:, 0].unique()
                logging.info("Bad batch sample indices:", bad_batch_indices.tolist())
                # optional: logging.info count of bad values per sample
                for idx in bad_batch_indices:
                    count = bad_mask[idx].sum().item()
                    logging.info(f"Sample {idx} has {count} invalid entries")
 
            #       logging.info("\n" + f"loss map {loss_map.mean()}")


            # MASK
            _, _, H, W = loss_map.shape
            mask = torch.zeros((1, 1, H, W), device=loss_map.device) # Make empty mask filled with 0s
            mask[..., :orig_h, :orig_w] = 1.0 # mark nonmasked areas as valid, using 1s
            mask = mask.expand(loss_map.size(0), -1, -1, -1) # expand over all batches

            # mask out padded areas
            loss_map = loss_map * mask
            # sum of valid pixels
            n_valid_pixels = mask.sum()
            #           logging.info("\n" + f"n valid pixels {n_valid_pixels}")
            # average only over valid pixels
            loss = loss_map.sum() / (n_valid_pixels + 1e-8)

            #       logging.info("\n" + f"loss mean {loss.mean()}")

            # BACKPROP
            # if loss.mean() < 0.001:
            #     torch.save(model.state_dict(),model_path,)
            #     logging.info("\n" + f"loss reached {loss} at epoch {epoch}. Exiting training. Saved model {model_path}")


            with torch.autograd.detect_anomaly():
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # STATS
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)
            
            # after optimizer.step()
            mean_w = mean_weight_value(model)
            loop.set_postfix(loss=loss_value, avg_loss=total_loss/(batch_idx+1), mean_weight=mean_w.item())


            if torch.isnan(loss):
                logging.error("\n" + "Encountered NaN loss — exiting.")
                exit()


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
        logging.info("\n" + f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


        torch.save(
            model.state_dict(),
            model_path,
        )

        # if abs(avg_loss - prev_loss) < 0.05:
        #    logging.info("improvment in loss was less than 2%. ending training")
        #    break
        # prev_loss = avg_loss




    logging.info("\n" + f"finished training. loss was {loss}. Exiting training. Saved model {model_path}")
    logging.info("\n" + "saved model")
    torch.save(
        model.state_dict(),
        model_path,
    )
    return  model_path




# ===



# logging.info("beginning to train", flush=True)

# batch_size = 32
# num_epochs = 100
# lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# for lr in lrs:
#     model_path = f"models/new_4x4_diffusion_{lr}.pth"
#     logging.info(f"training model {model_path}")
#     device="cuda"
#     image_size = 16
#     new_apply_diffusion(model_path, batch_size, image_size, num_epochs, lr, device)

# logging.info("done training all models", flush=True)
sys.stdout.flush()


