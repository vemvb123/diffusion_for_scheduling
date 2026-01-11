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



def adj_inference(proc_instance, model_path, n_samples, batch_size, n_features, model_path_enc, model_path_adj, dim_x=8, dim_y=8, pad_x=16, pad_y=16):
    device = "cuda"

    model_enc = deepinv.models.DiffUNet(
        in_channels=1, out_channels=n_features, pretrained=Path(model_path_enc)
    ).to(device)
    model_adj = deepinv.models.DiffUNet(
        in_channels=2, out_channels=n_features+1, pretrained=Path(model_path_adj)
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

    model_enc.eval()
    model_adj.eval()
    
    mask_value = 1
    x = None
    with torch.no_grad():
                # start timer
        start_time = time.perf_counter()


        allocations = torch.rand(batch_size, 1, 20, 20)
        allocations = allocations.to(device, dtype=torch.float32)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t
    
            enc_f = model_enc(proc_instance, t_tensor, type_t="timestep")
                
            model_input = torch.cat([
                    allocations,
                    enc_f,
                ], dim=1)

            pred_allocations = model_enc(model_input, t_tensor, type_t="timestep")
 
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
            if t > 0:
                noise = torch.randn_like(allocations)
            else:
                noise = torch.zeros_like(allocations) # ma ha maske??
                # noise = 0
        
            allocations = (1 / torch.sqrt(alpha)) * (
                allocations - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_allocations
            ) + torch.sqrt(beta) * noise

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    allocations = torch.clamp(allocations, 0, 1)

    return allocations, elapsed








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
