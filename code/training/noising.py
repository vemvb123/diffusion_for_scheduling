# ta vekk subset, gjor num epochs til 100


import torch


def get_noised_x(t, x, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    # Sample noise
    noise = torch.randn_like(x)

    noised = (
        sqrt_alphas_cumprod[t, None, None, None] * x
        + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
    )

    return noise, noised


def get_diffusion_schedule(beta_start, beta_end, timesteps, device = "cuda"):

    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
