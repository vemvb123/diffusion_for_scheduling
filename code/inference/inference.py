"""
inference.py contains code for running inference with trained models
"""


import logging

from code.inference.denoise import denoise_ddim, denoise_ddpm, get_inference_schedule

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


# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
def feature_inference_ddpm(proc_times, job_id, pos_job, model_path, n_samples, embed_size):

    device = "cuda"

    adj_model = deepinv.models.DiffUNet(
        in_channels=embed_size+1, out_channels=1, pretrained=Path(model_path)
    ).to(device)
    enc_model = deepinv.models.DiffUNet(
        in_channels=3, out_channels=embed_size, pretrained=Path(model_path)
    ).to(device)



    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    adj_model.eval()
    enc_model.eval()
    
    x = None
    with torch.no_grad():
        
        # creating a matrix, everything out side of the submatrix dim_x,dim_y has the value 1, while the matrix dim_x,dim_y has a random value.
        # similair to how x was masked during training
        x = torch.randn(n_samples, 1, 20, 20).to(device)
        # Set rows outside dim_x to 1
        #x[:, :, dim_x:, :] = 1
        # Set columns outside dim_y to 1 (for rows inside dim_x)
        #x[:, :, :dim_x, dim_y:] = 1

        # må fore inn maske...
        features = torch.cat([
            proc_times,
            job_id,
            pos_job,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        x = x.to(device, dtype=torch.float32)

        # start timer
        start_time = time.perf_counter()

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            f_enc = enc_model(features, t_tensor, type_t="timestep") 

            inputs = torch.cat([
                x,
                f_enc,
            ], dim=1)  # channels = 4

            predicted_noise = adj_model(inputs, t_tensor, type_t="timestep")
            x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed


# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
def adj_inference_ddpm(proc_times, job_id, pos_job, model_path, n_samples, h_when_masked, w_when_masked):

    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    model.eval()

    given_assignments = []
    
    x = None
    with torch.no_grad():
        
        # creating a matrix, everything out side of the submatrix dim_x,dim_y has the value 1, while the matrix dim_x,dim_y has a random value.
        # similair to how x was masked during training
        x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)
        # Set rows outside dim_x to 1
        #x[:, :, dim_x:, :] = 1
        # Set columns outside dim_y to 1 (for rows inside dim_x)
        #x[:, :, :dim_x, dim_y:] = 1

        # må fore inn maske...
        features = torch.cat([            
            proc_times,
            job_id,
            pos_job,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        x = x.to(device, dtype=torch.float32)

        # start timer
        start_time = time.perf_counter()

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                features,
            ], dim=1)  # channels = 4

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












def adj_inference_ddim(proc_times, job_id, pos_job, model_path, n_samples):

    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    model.eval()

    given_assignments = []


    eta = 0.5
    # create a sequence of DDIM timesteps if you want fewer steps
    # simple linear spacing (e.g. 50 steps out of 1000)
    ddim_steps = 700
    seq = list(torch.linspace(timesteps-1, 0, ddim_steps).long().to(device))    

    x = None
    with torch.no_grad():
        
        x = torch.randn(n_samples, 1, 20, 20).to(device)

        # må fore inn maske...
        features = torch.cat([            
            proc_times,
            job_id,
            pos_job,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        x = x.to(device, dtype=torch.float32)

        # start timer
        start_time = time.perf_counter()

        for i in range(len(seq)):
            t = seq[i]                           # current timestep
            t_prev = seq[i+1] if i+1 < len(seq) else -1  # next in sequence
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long) 

            inputs = torch.cat([
                x,
                features,
            ], dim=1)  # channels = 4

            predicted_noise = model(inputs, t_tensor, type_t="timestep")
            x = denoise_ddim(x, t, t_prev, alphas_cumprod, predicted_noise, eta)
            # x = get_denoised(t, alphas, alphas_cumprod, betas, predicted_noise)
           
            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments




# Ekperimentelt. vet ikke om funker onklig HELT.
# ====



# her kan jeg eks ta at ikke skal bruke en maskin som test.
# så kan jeg prøve på noe annet, eks øke brukbarhet, eller få ned inferencetid

# conditions er konkatinert av all conditioning data
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
