"""
inference.py contains code for running inference with trained models
"""


from diffusers import CosineDPMSolverMultistepScheduler, DDPMScheduler

from code.inference.denoise import denoise_ddpm, get_inference_schedule

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

import deepinv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.io as io
from torchvision import datasets, transforms

from tqdm import tqdm
import time

def zero_unused_slots(x, ops_ma_adj, valid_h, valid_w, device):

    x = x.to(device, dtype=torch.float32)
    ops_ma_adj = ops_ma_adj.to(device, dtype=torch.float32)
    mask = ops_ma_adj.any(dim=2, keepdim=True)  # collapse 24
    x = x * mask


    return x




def adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked, timesteps=1000, jump=None, cos=False, smart_init=False):
    
    device = "cuda"
    print(f"Using model {model_path}, with timesteps {timesteps}, and cos: {cos}")

    model = None
    optimizer = None

    try:
        model = deepinv.models.DiffUNet(
            in_channels=4,
            out_channels=1,
            pretrained=Path(model_path)
        ).to(device)

    except Exception as e:
        checkpoint = torch.load(model_path, map_location=device)

        model = deepinv.models.DiffUNet(in_channels=4, out_channels=1, pretrained=None)
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # define optimizer BEFORE loading its state
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    model.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="squaredcos_cap_v2",  # cosine schedule
        clip_sample=True,
        prediction_type="epsilon",
    )



    valid_h = 6   # valid rows
    valid_w = 55  # valid columns
    device = 'cuda'  # or 'cpu'



    given_assignments = []
    variation_over_time = []
    
    x = None
    with torch.no_grad():
        if smart_init == False:
            x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)
        # ===
        elif smart_init == True:
            # initialize tensor with zeros (full shape including padding)
            x = torch.zeros(n_samples, 1, h_when_masked, w_when_masked, device=device)

            # --- generate random row indices per sample and per valid column ---
            row_idx = torch.randint(0, valid_h, size=(n_samples, valid_w), device=device)

            # --- generate random normalized values per sample and per valid column ---
            values = torch.randint(1, 56, size=(n_samples, valid_w), device=device, dtype=torch.float32) / 55.0

            # --- column indices (valid columns only) ---
            col_idx = torch.arange(valid_w, device=device)

            # --- assign the values ---
            # x shape: [batch, 1, h, w], we assign using advanced indexing
            # we need to expand col_idx to match batch
            for b in range(n_samples):
                x[b, 0, row_idx[b], col_idx] = values[b]

        # ===

        # må fore inn maske...
        features = torch.cat([            
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        features = features.repeat(n_samples, 1, 1, 1)
        x = x.to(device, dtype=torch.float32)

        # start timer
        start_time = time.perf_counter()

        for t in reversed(range(timesteps)):
            if jump is not None and t == timesteps:
                t = jump
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                features,
            ], dim=1)

            predicted_noise = model(inputs, t_tensor, type_t="timestep")

            if cos:
                x_t = scheduler.step(predicted_noise, t, x).prev_sample
                diff = x_t[:, :, :6, :60] - x[:, :, :6, :60]
                if t == 0 or t == 10 or t == 25 or t == 50 or t == 75 or t ==t ==  99:
                    variation = diff.abs().sum(dim=-1).squeeze(1)
                    variation_over_time.append(variation)
                x = x_t
            else:
                x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

            # if t % 10 == 0:
            if t == 0:
                given_assignments.append(x.clone())

            """
            if t % 100==0:
                given_assignments.append(x.clone())
            """

            x = zero_unused_slots(x, ops_ma_adj, valid_h, valid_w, device)

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments, variation_over_time



