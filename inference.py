"""
inference.py contains code for running inference with trained models
"""


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

# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
def feature_inference(proc_times, job_id, pos_job, model_path, n_samples, embed_size):

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

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    model.eval()
    
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




# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
def adj_inference(proc_times, job_id, pos_job, model_path, n_samples):

    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
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

    given_assignments = []
    
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

            inputs = torch.cat([
                x,
                features,
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
            
            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    return x, elapsed, given_assignments














# her kan jeg eks ta at ikke skal bruke en maskin som test.
# så kan jeg prøve på noe annet, eks øke brukbarhet, eller få ned inferencetid

def guiding_function(x): #  is batch of instances
    # x has shape [1, 1, 20, 20]
    # extract the 4×16 region
    x_inside = x[:, :, :4, :16]   # shape [1, 1, 4, 16]

    # select the bottom row of that 4×16 → index 3
    bottom_row = x_inside[:, :, 3, :]  # shape [1, 1, 16]

    # compute mean absolute value of bottom row
    error = torch.abs(bottom_row).mean()

    return error


# conditions er konkatinert av all conditioning data
def guide_adj_inference(conditions, n_channels, model_path, n_to_make, batch_size):
    device = "cuda"
    guidence_scale = 0.25

    model = deepinv.models.DiffUNet(
        in_channels=n_channels, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Amount of trainable parameters: {trainable_params}")



    conditions = conditions.to(device, dtype=torch.float32)


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

    # start timer
    start_time = time.perf_counter()

    x_allocations = torch.rand(batch_size, 1, 20, 20)
    x_allocations = x_allocations.to(device, dtype=torch.float32)

    allocations_over_time = []
    errors = []
    for t in reversed(range(timesteps)):
        t_tensor = torch.ones(n_to_make, device=device).long() * t



        model_input = torch.cat([
                x_allocations,
                conditions,
            ], dim=1)
        
        x_allocations = x_allocations.detach().requires_grad_(True)    
        pred_x_allocations = model(model_input, t_tensor, type_t="timestep")

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]


        estimated_x0 = (
            (x_allocations - torch.sqrt(1 - alpha_cumprod) * pred_x_allocations) /
            torch.sqrt(alpha_cumprod)
        )

        guidance_loss = guiding_function(estimated_x0) * guidence_scale 
        # compute gradient wrt x_allocations
        grad_x = torch.autograd.grad(guidance_loss, x_allocations)[0]
        # update the noised sample towards lower guidance loss
        x_guided = x_allocations.detach() - guidence_scale * grad_x




        # skal jeg bruke predicted noise? eller nei, det er vell bare for shape... man x er jo her med cond, så må kanskje endre, så lik predicted noise
        if t > 0:
            noise = torch.randn_like(x_allocations)
        else:
            noise = torch.zeros_like(x_allocations) # ma ha maske??
            # noise = 0
    
        x_prev = (1 / torch.sqrt(alpha)) * (
            x_guided - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_x_allocations # sto tidligere x_allocations
        ) + torch.sqrt(beta) * noise

        x_allocations = x_prev.detach()

        if t % 100 == 0:
            allocations_over_time.append(x_allocations.clone())
            errors.append(guidance_loss)

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x_allocations = torch.clamp(x_allocations, 0, 1)

    return x_allocations, elapsed, allocations_over_time, errors








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
