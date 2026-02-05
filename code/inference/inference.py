"""
inference.py contains code for running inference with trained models
"""



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



def adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked):

    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")


    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    model.eval()

    given_assignments = []
    
    x = None
    with torch.no_grad():
        
        x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)

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
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                features,
            ], dim=1)

            predicted_noise = model(inputs, t_tensor, type_t="timestep")

            x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments



