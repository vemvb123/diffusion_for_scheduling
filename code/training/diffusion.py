"""
diffusion.py contains code for training diffusion models
"""
from diffusers import CosineDPMSolverMultistepScheduler, DDPMScheduler

import os
import logging

from code.training.experimental import run_epoch_feature, run_epoch_penalty_feasibility
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
from diffusers import UNet1DModel

import matplotlib.pyplot as plt








def run_epoch(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode, valid_h, valid_w, 
            scheduler=None, model_name=None, epoch_num=None,
            loss_plot_folder=None):


    total_loss = 0.0
    batch_losses = []

    fig, ax = plt.subplots()
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss {mode}")

    if mode == "train":
        model_adj.train()
    else:
        model_adj.eval()


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
            optimizer.zero_grad(set_to_none=True)

            pred = model_adj(model_input, t, type_t="timestep")

            pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)
            loss = nn.MSELoss()(pred_valid, noise_valid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_adj.parameters(), 1.0)
            optimizer.step()

        else:
            with torch.no_grad():
                pred = model_adj(model_input, t, type_t="timestep")

                pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)
                loss = nn.MSELoss()(pred_valid, noise_valid)


        loss_value = loss.item()
        total_loss += loss_value


        batch_losses.append(loss_value)

        loop.set_postfix(loss=loss_value)


        # plotting loss
        ax.clear()
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss {mode} (updated) epoch nr {epoch_num}")
        ax.plot(batch_losses, color="blue")
        # Save figure to disk as PNG (overwrite each batch)
        if model_name is not None and loss_plot_folder is not None:
            fig.savefig(f"{loss_plot_folder}/{mode}_{model_name}_{epoch_num}.png")

 
    plt.close(fig)

    return loop, model_adj, total_loss / len(loop)





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
    penalty=False,
    beta_start=1e-4, beta_end=0.02
):
    n_base_features = train_dataset.n_base_features

    # For linear schedule
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule(beta_start, beta_end, timesteps)

    # For cosine schedule
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

    # f is experimental
    if model_type != "adj" and model_type != "f":
        raise ValueError(f"model_type must be either adj or f .kk. but value was #{model_type}#")

    # running different diffusion algorithms depening on the model type
    run_epoch_func = None
    if model_type == "adj":
        run_epoch_func = run_epoch
    elif model_type == "f": # experimental
        run_epoch_func = run_epoch_feature 
    if penalty: # experimental
        run_epoch_func = run_epoch_penalty_feasibility

    train_loader, test_loader = get_dataset_loaders(train_dataset, test_dataset, batch_size=batch_size) # subset=True ... for testing/debugging with little data

    logging.info(f"N instances in train dataset: { len(train_loader.dataset) }")
    logging.info(f"N instances in test dataset: { len(test_loader.dataset) }")
    
    model_adj, model_enc, optimizer = get_models(model_type, n_base_features, n_embed_features, lr, path=model_path_adj)

    trainable_params = sum(p.numel() for p in model_adj.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")

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


        model_name = model_path_adj.split('/')[-1].split('.')[0]
        # Running on train set
        train_loop, model_adj, avg_loss = run_epoch_func(
            train_loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            "train", valid_h, valid_w, scheduler, model_name, epoch_num=epoch, 
            loss_plot_folder=graph_save_folder
        )
        
        # Running on train set
        test_loop, model_adj, avg_loss_test = run_epoch_func(
            test_loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            "test", valid_h, valid_w, scheduler, model_name, epoch_num=epoch,
            loss_plot_folder=graph_save_folder
        )


        # log losses
        losses_folder = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/losses"
        np.save(
            f"{losses_folder}/losses_epoch_{epoch+1}.npy",
            np.array(all_losses)
        )
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {avg_loss:.4f}, Test Loss: {avg_loss_test:.4f}"
        )

        all_losses.append(avg_loss)
        all_losses_test.append(avg_loss_test)

        # save model for current epoch
        os.makedirs(os.path.dirname(model_path_adj), exist_ok=True)
        torch.save({ 
                'model_state_dict': model_adj.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), }, 
            model_path_adj)
        

        if model_type == "f":
            torch.save(model_enc.state_dict(), model_enc)
        
        # Stopping training early, if loss stops decreasing
        if len(all_losses_test) >= 4:
            if all_losses_test[-1] > all_losses_test[-4]:
                logging.info("Test loss has not gone down for 4 epochs - stopping early")
                break
    
    # saving final models
    torch.save({ 
                'model_state_dict': model_adj.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), }, 
            model_path_adj)

    if model_type == "f":
        torch.save(model_enc.state_dict(), model_enc)

    logging.info(f"done training. Saved model {model_path_adj}")

    # return model path and last obtained loss
    return None, model_path_adj, all_losses[-1]



