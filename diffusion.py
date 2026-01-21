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
# import new_dataset

import sys

from diffusers import UNet1DModel




'''
dataset has: image of machine relations (1 for each machinbe)
model outputs: set of coordinates, to one of the coordinates for an operation, where each coordinate in sequence gives the assignment order
target: ideal coordinates
'''



'''
Hvordan feature vectors...

Prosseseringstid:
    sender inn adj med prosseseringstidene (b, 1, w, h)
    bruker output channels til å få flere dimensjoner (b, 256, w, h)
    Flater, ved wh, så hele h inneholder alle nodene, og flater ut langs 256 som blir w
    så shape blir (b, 1, 256, w x h)

    ID tror jeg ikke er verdt noe, siden en ma eller op kan endre seg for hver instanse, så vil ikke en op eller ma bety det samme.
    Men generelt for features, eks mengde precessors osv, kan man ha en adj matrise, så prossesere det likt som med prosseseringstid.

'''
from torch.utils.data import DataLoader, Subset
# ta vekk subset, gjor num epochs til 100



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




def run_epoch_feature(loop, device, timesteps,
            model_enc, model_adj, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode):
    total_loss = 0.0
    for batch_idx, (target_assignments, proc_times, job_id, pos_job) in enumerate(loop): 
        """ 
        logging.info("dataset shapes")
        logging.info(proc_times.shape)
        logging.info(job_id.shape)
        logging.info(pos_job.shape)
        """

        features = torch.cat([
            proc_times,
            job_id,
            pos_job,
        ], dim=1)


        # logging.info(f"f shape {features.shape}")
        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue
 

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device) 
        # logging.info(f't shape {t.shape}')

        # encode images
        optimizer.zero_grad()
        enc_f = model_enc(features, t, type_t="timestep")
        # logging.info(f"encoded f {enc_f.shape}")
        
        # Sample noise
        noise = torch.randn_like(target_assignments)
        # Apply forward diffusion process at timestep t
        noised = (
            sqrt_alphas_cumprod[t, None, None, None] * target_assignments
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        #logging.info("catination")
        #logging.info(noise.shape)
        #logging.info(enc_f.shape)

        model_input = torch.cat([
                noised,
                enc_f,
            ], dim=1)
        
        #logging.info(model_input.shape)

        # Predict noise
        pred = model_adj(model_input, t, type_t="timestep")
        #logging.info(f"made prediction, {pred.shape}")
        
        # only considering the valid loss region
        valid_h = 4
        valid_w = 16
        pred_valid  = pred[..., :valid_h, :valid_w]
        noise_valid = noise[..., :valid_h, :valid_w]
        loss = nn.MSELoss()(pred_valid,noise_valid)
        # loss = nn.MSELoss()(pred, noise)
        if mode == "train":
            loss.backward()
            optimizer.step()

        # Save the loss value
        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, model_enc, total_loss / len(loop)






def feature_diffusion(loss_image_path, train_dataset, test_dataset, ordered: int, n_base_features: int, n_embed_features: int,
                      model_path_enc: str, model_path_adj: str, graph_name: str, graph_save_folder: str, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda", batch_size: int = 32):



    #train_subset_dataset = Subset(train_dataset, range(32*5))
    #test_subset_dataset = Subset(test_dataset, range(32*5))
    #train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=False)
    #test_loader = DataLoader(test_subset_dataset, batch_size=32, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")
    
    # out channels, flere, samme som embedding dimensjon .. men burde kanskje endre bilderep, fordi nu sliter koordinatet
    model_enc = deepinv.models.DiffUNet(in_channels=n_base_features, out_channels=n_embed_features, pretrained=None).to(device)
    model_adj = deepinv.models.DiffUNet(in_channels=n_embed_features+1, out_channels=1, pretrained=None).to(device)
    # model_coords = UNet1DModel(in_channels=n_features+1, out_channels=1).to(device)

    # printing amount of parameters
    trainable_params = sum(p.numel() for p in model_enc.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    # optimizer_img = torch.optim.Adam(model_enc.parameters(), lr=lr)
    # optimizer_coords = torch.optim.Adam(model_coords.parameters(), lr=lr)

    optimizer = torch.optim.Adam(
        list(model_enc.parameters()) + list(model_adj.parameters()),
        lr=lr
    )

    mse = deepinv.loss.MSE()

    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    all_losses_test = []
    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", unit="batch")
        test_loop = tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{num_epochs}", unit="batch")

        train_loop, model_adj, model_enc, avg_loss = run_epoch_feature(train_loop, device, timesteps,
                                                        model_enc, model_adj, optimizer,
                                                        batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode="train")
        
        test_loop, model_adj, model_enc, avg_loss_test = run_epoch_feature(test_loop, device, timesteps,
                                                        model_enc, model_adj, optimizer,
                                                        batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode="test")
        


       # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}_{graph_name}.npy", np.array(all_losses))
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_loss_test:.4f}")

        all_losses.append(avg_loss)
        all_losses_test.append(avg_loss_test)

        torch.save(model_enc.state_dict(),model_path_enc,)
        torch.save(model_adj.state_dict(),model_path_adj,)


        if len(all_losses_test) >= 4:
            if all_losses_test[-1] > all_losses_test[-4]:
                logging.info(
                    "Test loss has not gone down for 4 epochs - stopping early"
                )
                break
    


    logging.info("saved loss image")
    plot_losses(graph_save_folder, f"{graph_name} train", all_losses)
    plot_losses(graph_save_folder, f"{graph_name} test", all_losses_test)

    logging.info("saved model")
    torch.save(model_enc.state_dict(),model_path_enc,)
    torch.save(model_adj.state_dict(),model_path_adj,)

    return model_path_enc, model_path_adj, all_losses[-1]





def run_epoch_adj(
    loop, device, timesteps,
    model_adj, optimizer,
    batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
    mode
):
    total_loss = 0.0

    for batch_idx, (target_assignments, proc_times, job_id, pos_job) in enumerate(loop):

        features = torch.cat([
            proc_times,
            job_id,
            pos_job,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)


        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue
                

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(target_assignments)

        # Apply forward diffusion
        # TODO print the instance idx
        # TODO ignore instances which gives the error
        noised = (
            sqrt_alphas_cumprod[t, None, None, None] * target_assignments
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model_adj(model_input, t, type_t="timestep")

        valid_h = 4
        valid_w = 16
        pred_valid  = pred[..., :valid_h, :valid_w]
        noise_valid = noise[..., :valid_h, :valid_w]

        loss = nn.MSELoss()(pred_valid, noise_valid)

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)


def adj_diffusion(
    loss_image_path,
    train_dataset,
    test_dataset,
    ordered: int,
    n_base_features: int,
    n_embed_features: int,
    model_path_enc: str,      # keep this for signature match
    model_path_adj: str,
    graph_name: str,
    graph_save_folder: str,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    batch_size: int = 32
):


    #train_subset_dataset = Subset(train_dataset, range(32*5))
    #test_subset_dataset = Subset(test_dataset, range(32*5))
    #train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=False)
    #test_loader = DataLoader(test_subset_dataset, batch_size=32, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")
    

    model_adj = deepinv.models.DiffUNet(
        in_channels=n_base_features + 1,
        out_channels=1,
        pretrained=None
    ).to(device)

    trainable_params = sum(p.numel() for p in model_adj.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")

    optimizer = torch.optim.Adam(model_adj.parameters(), lr=lr)

    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    all_losses = []
    all_losses_test = []

    for epoch in range(num_epochs):

        train_loop = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch+1}/{num_epochs}",
            unit="batch"
        )
        test_loop = tqdm(
            test_loader,
            desc=f"Test Epoch {epoch+1}/{num_epochs}",
            unit="batch"
        )

        train_loop, model_adj, avg_loss = run_epoch_adj(
            train_loop, device, timesteps,
            model_adj, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            mode="train"
        )

        test_loop, model_adj, avg_loss_test = run_epoch_adj(
            test_loop, device, timesteps,
            model_adj, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            mode="test"
        )

        np.save(
            f"./losses/losses_epoch_{epoch+1}.npy",
            np.array(all_losses)
        )

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {avg_loss:.4f}, Test Loss: {avg_loss_test:.4f}"
        )

        all_losses.append(avg_loss)
        all_losses_test.append(avg_loss_test)

        torch.save(model_adj.state_dict(), model_path_adj)

        if len(all_losses_test) >= 4:
            if all_losses_test[-1] > all_losses_test[-4]:
                logging.info(
                    "Test loss has not gone down for 4 epochs - stopping early"
                )
                break
    

    logging.info("saved loss image")
    plot_losses(graph_save_folder, f"{graph_name} train", all_losses)
    plot_losses(graph_save_folder, f"{graph_name} test", all_losses_test)

    torch.save(model_adj.state_dict(), model_path_adj)

    return None, model_path_adj, all_losses[-1]



















