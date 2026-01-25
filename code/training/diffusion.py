"""
diffusion.py contains code for training diffusion models
"""



import logging

from training.scheduling import get_diffusion_schedule, get_noised_x
from training.utils import get_dataset_loaders, get_models, mask_invalid, plot_losses

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

import torch
import numpy as np

import torch.nn as nn

import torch.nn as nn
# import new_dataset


from diffusers import UNet1DModel
from deepinv.models.diffunet import DiffUNet



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


def run_epoch_feature(loop, device, timesteps,
            model_adj, model_enc, optimizer,
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
        noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
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

        pred_valid, noise_valid = mask_invalid(4, 16, pred, noise)
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






def run_epoch_adj(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode):


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
        noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model_adj(model_input, t, type_t="timestep")

        pred_valid, noise_valid = mask_invalid(4, 16, pred, noise)
        loss = nn.MSELoss()(pred_valid, noise_valid)

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)






def diffusion(
    model_type: str, # must be either "adj" for adjecency model or "f" for feature vector model
    loss_image_path,
    train_dataset,
    test_dataset,
    n_base_features: int,
    n_embed_features: int,
    model_path_adj: str,
    model_path_enc: str,      # keep this for signature match
    graph_name: str,
    graph_save_folder: str,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    batch_size: int = 32
):

    if model_type != "adj" or model_type != "f":
        raise ValueError(f"model_type must be either adj or f .. but value was {model_type}")

    # running different diffusion algorithms depening on the model type
    run_epoch_func = None
    if model_type == "adj":
        run_epoch_func = run_epoch_adj
    elif model_type == "f":
        run_epoch_func = run_epoch_feature 

    train_loader, test_loader = get_dataset_loaders(train_dataset, test_dataset, batch_size=batch_size)
    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")
    
    model_adj, model_enc, optimizer = get_models(model_type, n_base_features, n_embed_features, lr)

    trainable_params = sum(p.numel() for p in model_adj.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")

    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule(beta_start, beta_end, timesteps)

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


        train_loop, model_adj, avg_loss = run_epoch_func(
            train_loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            mode="train"
        )

        test_loop, model_adj, avg_loss_test = run_epoch_func(
            test_loop, device, timesteps,
            model_adj, model_enc, optimizer,
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
        if model_type == "f":
            torch.save(model_enc.state_dict(), model_enc)

        if len(all_losses_test) >= 4:
            if all_losses_test[-1] > all_losses_test[-4]:
                logging.info("Test loss has not gone down for 4 epochs - stopping early")
                break
    

    logging.info("saved loss image")
    plot_losses(graph_save_folder, f"{graph_name} train", all_losses)
    plot_losses(graph_save_folder, f"{graph_name} test", all_losses_test)

    torch.save(model_adj.state_dict(), model_path_adj)
    if model_type == "f":
        torch.save(model_enc.state_dict(), model_enc)

    return None, model_path_adj, all_losses[-1]



