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
import utils
from torchvision import datasets, transforms

from tqdm import tqdm
import time



def adj_inference(proc_instance, model_path, n_samples, dim_x=8, dim_y=8, pad_x=16, pad_y=16):
    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=2, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()
    
    mask_value = 1
    x = None
    with torch.no_grad():
        
        # creating a matrix, everything out side of the submatrix dim_x,dim_y has the value 1, while the matrix dim_x,dim_y has a random value.
        # similair to how x was masked during training
        x = torch.randn(n_samples, 1, pad_x, pad_y).to(device)
        # Set rows outside dim_x to 1
        x[:, :, dim_x:, :] = 1
        # Set columns outside dim_y to 1 (for rows inside dim_x)
        x[:, :, :dim_x, dim_y:] = 1

        # må fore inn maske...
        proc_instance = proc_instance.unsqueeze(0).unsqueeze(0).to(device)

        model_input = torch.cat([
                x,
                proc_instance,
            ], dim=1)

        # start timer
        start_time = time.perf_counter()

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                proc_instance,
            ], dim=1)  # channels = 4

            predicted_noise = model(inputs, t_tensor, type_t="timestep")
                
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # ma ha maske??
                # noise = 0
        
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed








"""
logging.info("running")
# model_path = "diffusionAsPlugAndPlay_lr_masks_0.0001.pth"
n_samples = 1
beta_start = 0.0001
# file_to_solve = "instances/4ma_2op_2j_80000"
model_path = "models/diffusion_mnist_guided.pth"
apply_conditional_inference(32, 32, model_path, n_samples, beta_start)
# apply_inference_existing_problem_test(file_to_solve, 64, 64, model_path, n_samples, beta_start)
logging.info("done")
"""
