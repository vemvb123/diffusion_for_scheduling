import deepinv
import time
from pathlib import Path
import torch

from code.inference.denoise import get_inference_schedule




def similair_MU(x):
    pass




def guiding_function(x): #  is batch of instances
    # x has shape [1, 1, 20, 20]
    # extract the 4×16 region
    x_inside = x[:, :, :4, :16]   # shape [1, 1, 4, 16]

    # select the bottom row of that 4×16 → index 3
    bottom_row = x_inside[:, :, 3, :]  # shape [1, 1, 16]

    # compute mean absolute value of bottom row
    error = torch.abs(bottom_row).mean()

    return error






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
    betas, alphas, alphas_cumprod, = get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

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