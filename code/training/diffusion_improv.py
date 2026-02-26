"""
diffusion.py contains code for training diffusion models
"""
from diffusers import CosineDPMSolverMultistepScheduler, DDPMScheduler
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import logging

from code.training.experimental import run_epoch_feature
from code.training.noising import get_diffusion_schedule, get_noised_x
from code.training.utils import get_dataset_loaders, get_models, mask_invalid, plot_losses

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




def run_epoch(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode, valid_h, valid_w, 
            scheduler=None):


    total_loss = 0.0

    for batch_idx, (target_assignments, proc_times, job_ops_adj, ops_ma_adj) in enumerate(loop):

        features = torch.cat([
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)

        noise, noised = None, None
        if scheduler == None:
            noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        else:
            noise = torch.randn_like(target_assignments)
            noised = scheduler.add_noise(target_assignments, noise, t)


        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model_adj(model_input, t, type_t="timestep")

        pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)
        loss = nn.MSELoss()(pred_valid, noise_valid)

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)


from code.training.experimental import run_epoch_penalty_feasibility

def diffusion(
    model_type: str, # must be either "adj" for adjecency model or "f" for feature vector model
    train_dataset,
    test_dataset,
    n_embed_features: int,
    model_path_adj: str,
    model_path_enc: str,      # keep this for signature match
    graph_name: str,
    graph_save_folder: str,
    valid_h, valid_w,
    timesteps: int = 1000,
    scheduler_timesteps: int = 1000,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
    batch_size: int = 32,
    use_cos=True,
    penalty=False
):
    n_base_features = train_dataset.n_base_features

    beta_start = 1e-4
    beta_end = 0.02
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule(beta_start, beta_end, timesteps)


    scheduler = None
    if use_cos:
        scheduler = DDPMScheduler(
            num_train_timesteps=scheduler_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="squaredcos_cap_v2",  # cosine schedule
            clip_sample=True,
            prediction_type="epsilon",
        )


    epoch_func = None
    if penalty:
        epoch_func = run_epoch_penalty_feasibility
    else:
        epoch_func = run_epoch

    if model_type != "adj" and model_type != "f":
        raise ValueError(f"model_type must be either adj or f .kk. but value was #{model_type}#")

    train_loader, test_loader, val_loader = get_dataset_loaders(train_dataset, test_dataset, batch_size=batch_size, val_ratio=0.2) # subset=True ... for testing med subset

    model_adj, model_enc, optimizer = get_models(model_type, n_base_features, n_embed_features, lr)

    trainable_params = sum(p.numel() for p in model_adj.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    all_losses = []
    all_losses_val = []

    for epoch in range(num_epochs):

        train_loop = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch+1}/{num_epochs}",
            unit="batch"
        )
        val_loop = tqdm(
            val_loader,
            desc=f"Validation Epoch {epoch+1}/{num_epochs}",
            unit="batch"
        )

        train_loop, model_adj, avg_loss = epoch_func(
            train_loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            "train", valid_h, valid_w, scheduler
        )

        val_loop, model_adj, avg_loss_val = epoch_func(
            val_loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            "test", valid_h, valid_w, scheduler
        )



        losses_folder = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/losses"
        np.save(
            f"{losses_folder}/losses_epoch_{epoch+1}.npy",
            np.array(all_losses)
        )

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {avg_loss:.4f}, Val Loss: {avg_loss_val:.4f}"
        )

        all_losses.append(avg_loss)
        all_losses_val.append(avg_loss_val)

        torch.save(model_adj.state_dict(), model_path_adj)
        if model_type == "f":
            torch.save(model_enc.state_dict(), model_enc)

        if len(all_losses_val) >= 4:
            if all_losses_val[-1] > all_losses_val[-4]:
                logging.info("Validation loss has not gone down for 4 epochs - stopping early")
                break
    

    # Running on test set
    test_loop = tqdm( test_loader, desc=f"Running on test set", unit="batch" )
    test_loop, model_adj, avg_loss_test = run_epoch(
        test_loop, device, timesteps,
        model_adj, model_enc, optimizer,
        batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        "test", valid_h, valid_w, scheduler
    )
    logging.info(f"Average test loss: {avg_loss_test}")


    # Saving graphs from training
    logging.info("saved loss image")
    plot_losses(graph_save_folder, f"{graph_name} train", all_losses)
    plot_losses(graph_save_folder, f"{graph_name} validation", all_losses_val)

    torch.save(model_adj.state_dict(), model_path_adj)
    if model_type == "f":
        torch.save(model_enc.state_dict(), model_enc)
    logging.info(f"done training. Saved model {model_path_adj}")
    return None, model_path_adj, all_losses_val[-1]



