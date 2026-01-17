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





def guiding_function(x): #  is batch of instances
    target = 0
    error = torch.abs(x - target).mean()
    return error


# conditions er konkatinert av all conditioning data
def guide_adj_inference(conditions, n_channels, model_path, n_to_make, batch_size):
    device = "cuda"
    guidence_scale = 0.25

    model = deepinv.models.DiffUNet(
        in_channels=n_channels, out_channels=1, pretrained=Path(model_path)
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
    
    with torch.no_grad():
                # start timer
        start_time = time.perf_counter()

        x_allocations = torch.rand(batch_size, 1, 20, 20)
        x_allocations = x_allocations.to(device, dtype=torch.float32)

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_to_make, device=device).long() * t
    
            model_input = torch.cat([
                    x_allocations,
                    conditions,
                ], dim=1)
            
            x_allocations = x_allocations.detach().requires_grad_()
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
                x_allocations - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_x_allocations
            ) + torch.sqrt(beta) * noise

            x_allocations = x_prev.detach()




    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x_allocations = torch.clamp(x_allocations, 0, 1)

    return x_allocations, elapsed








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
