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




def run_epoch(loop, device, timesteps,
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



    #subset_dataset = Subset(dataset, range(32*5))
    #loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)

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
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        """
        
        """

        train_loop, model_adj, model_enc, avg_loss = run_epoch(train_loop, device, timesteps,
                                                        model_enc, model_adj, optimizer,
                                                        batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode="train")
        
        test_loop, model_adj, model_enc, avg_loss_test = run_epoch(test_loop, device, timesteps,
                                                        model_enc, model_adj, optimizer,
                                                        batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode="test")
        

        """
        
        """



       # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch+1}_{graph_name}.npy", np.array(all_losses))

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_loss_test:.4f}")

        all_losses.append(avg_loss)
        all_losses_test.append(avg_loss_test)

        torch.save(model_enc.state_dict(),model_path_enc,)
        torch.save(model_adj.state_dict(),model_path_adj,)


        # If we have at least 2 epochs, check the difference
        if len(all_losses) >= 3:
            if all_losses[-1] > all_losses[-2] and all_losses[-2] > all_losses[-3]:
                logging.info(
                    f"Loss has increased for two consecutive epochs "
                    "- stopping training early."
                )
                break


    logging.info("saved loss image")
    plot_losses(graph_save_folder, graph_name, all_losses)
    plot_losses(graph_save_folder, graph_name, all_losses_test)

    logging.info("saved model")
    torch.save(model_enc.state_dict(),model_path_enc,)
    torch.save(model_adj.state_dict(),model_path_adj,)

    return model_path_enc, model_path_adj, all_losses[-1]






def adj_diffusion(loss_image_path, dataset, ordered: int, n_base_features: int, n_embed_features: int,
                      model_path_enc: str, model_path_adj: str, num_epochs: int = 5, lr: float = 1e-3, device: str = "cuda", batch_size: int = 32):

    #subset_dataset = Subset(dataset, range(32*5))
    #loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    logging.info(f"N instances in dataset: { len(loader.dataset) }")

    # out channels, flere, samme som embedding dimensjon .. men burde kanskje endre bilderep, fordi nu sliter koordinatet
    model_adj = deepinv.models.DiffUNet(in_channels=n_base_features + 1, out_channels=1, pretrained=None).to(device)
    # model_coords = UNet1DModel(in_channels=n_features+1, out_channels=1).to(device)

    # printing amount of parameters
    trainable_params = sum(p.numel() for p in model_adj.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")

    # optimizer_img = torch.optim.Adam(model_enc.parameters(), lr=lr)
    # optimizer_coords = torch.optim.Adam(model_coords.parameters(), lr=lr)

    optimizer = torch.optim.Adam(model_adj.parameters(), lr=lr)

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
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (target_assignments, proc_times, job_id, pos_job) in enumerate(loop):
            
            features = torch.cat([
                proc_times,
                job_id,
                pos_job,
            ], dim=1)

            #logging.info(f"f shape {features.shape}")
            features = features.to(device, dtype=torch.float32)
            target_assignments = target_assignments.to(device, dtype=torch.float32)

            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            #logging.info(f't shape {t.shape}')


            # Sample noise
            noise = torch.randn_like(target_assignments)
            # Apply forward diffusion process at timestep t
            noised = (
                    sqrt_alphas_cumprod[t, None, None, None] * target_assignments
                    + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )

            #logging.info("catination")
            #logging.info(noise.shape)
            #logging.info(features.shape)
            model_input = torch.cat([
                noised,
                features,
            ], dim=1)


            #logging.info(model_input.shape)

            # Predict noise
            optimizer.zero_grad()
            pred = model_adj(model_input, t, type_t="timestep")
            #logging.info(f"made prediction, {pred.shape}")

            valid_h = 4
            valid_w = 16
            pred_valid  = pred[..., :valid_h, :valid_w]
            noise_valid = noise[..., :valid_h, :valid_w]
            loss = nn.MSELoss()(pred_valid,noise_valid)
            # loss = nn.MSELoss()(pred, noise)
            loss.backward()
            optimizer.step()
            #logging.info("one looped")

            # Save the loss value
            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

            loop.set_postfix(loss=loss_value)

        # Save all losses from this epoch
        # Create directories if they don't exist
        os.makedirs("./losses", exist_ok=True)
        os.makedirs("./weights", exist_ok=True)
        os.makedirs("./img/noised", exist_ok=True)
        os.makedirs("./img/denoised", exist_ok=True)
        os.makedirs("./img/original", exist_ok=True)

        # all_losses.extend(epoch_losses)

        # Save the losses list after each epoch
        np.save(f"./losses/losses_epoch_{epoch + 1}.npy", np.array(all_losses))

        avg_loss = total_loss / len(loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(
            model_adj.state_dict(),
            model_path_adj,
        )

        all_losses.append(avg_loss)

        # If we have at least 2 epochs, check the difference
        if len(all_losses) >= 3:
            if all_losses[-1] > all_losses[-2] and all_losses[-2] > all_losses[-3]:
                logging.info(
                    f"Loss has increased for two consecutive epochs "
                    "- stopping training early."
                )
                break

        # If we have at least 2 epochs, check the difference
        if len(all_losses) >= 2:
            diff = abs(all_losses[-2] - all_losses[-1])
            if diff < 0.0001:
                logging.info(
                    f"Loss change {diff:.6f} < 0.0001 "
                    "- stopping training early."
                )
                break
        



    logging.info("saved model")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(all_losses)+1), all_losses, marker='o')
    plt.title(f"Training Loss per Epoch, model {model_path_adj}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # save plot to file
    plt.savefig(f"{loss_image_path}/adj_{ordered}_loss_over_epochs.png")
    logging.info("saved loss image")

    torch.save(
        model_adj.state_dict(),
        model_path_adj,
    )

    return None, model_path_adj, model_adj, epoch_losses[-1]





# DIFFUSION MED BILDER OG  KOORDINATER
def flatten_feature_vector(x):
    x = x.reshape(x.size(0), x.size(1), -1)  # (b, f, w*h)
    x = x.permute(0, 2, 1).unsqueeze(1)       # (b, 1, w*h, f)

    return x


def diffusion(dataset, op_n, model_path: str, batch_size: int = 32, num_epochs: int = 100, lr: float = 1e-3, device: str = "cuda"):
    logging.info("kjører diffusion")


    loader = DataLoader(dataset, batch_size=32, shuffle=True)



    logging.info(f"Instances in dataset: { len(loader.dataset) }")
    logging.info(f"Instances in loader: { len(loader) }")

    # out channels, flere, samme som embedding dimensjon .. men burde kanskje endre bilderep, fordi nu sliter koordinatet
    model_img = deepinv.models.DiffUNet(in_channels=1, out_channels=16, pretrained=None).to(device)
    model_coords = UNet1DModel(in_channels=4, out_channels=3).to(device)

    # printing amount of parameters
    trainable_params = sum(p.numel() for p in model_img.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    optimizer_img = torch.optim.Adam(model_img.parameters(), lr=lr)
    optimizer_coords = torch.optim.Adam(model_coords.parameters(), lr=lr)

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
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []  # Store losses for this epoch
        # Instances: the different images, shape (amt images, w, h, b)
        # coords: coordinates, shape (amt cords (3 for x,y,ma), w, b)
        # (imgs1, imgs2, imgs3, imgs4), (col1, col2, col3) = batch
        for batch_idx, (imgs, coords) in enumerate(loader):
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
            noised = (
                sqrt_alphas_cumprod[t, None, None, None] * coords_cat
                + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )


            # TODO: flatut enkoda bilder
            # TODO skjekk om data ser riktig ut
            encoded_flat = enc_imgs.flatten(start_dim=2)

            # TODO Concat encoda bilder, og koordinater
            # TODO skjekk at shape blir riktig
            logging.info("catination")
            logging.info(noised.shape)
            logging.info(encoded_flat.shape)


            model_input = torch.cat([
                    noised,
                    encoded_flat,
                ], dim=1)

            logging.info(model_input.shape)
            return

            # TODO send encodete bilder og koordinater inn i modell, fa tilbake koordinater
            # Predict noise
            pred = model_coords(pred, t, type_t="timestep")

            # TODO regn lossm og backpropagate, for 1d og 2d .. hvordan ta loss fra en modell, og gi til den andre..
            loss = nn.MSELoss()(pred, noise)
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

        avg_loss = total_loss / len(loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(
            model_coords.state_dict(),
            model_path,
        )


    logging.info("saved model")
    torch.save(
        model_coords.state_dict(),
        model_path,
    )
    return  model_path

