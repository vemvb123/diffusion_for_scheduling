import torch


def denoise_ddim(x, t, t_prev, alphas_cumprod, predicted_noise, eta):
    # 2) Compute predicted clean data estimate
    alpha_t = alphas_cumprod[t]
    pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

    # 3) Compute deterministic DDIM update
    alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else alphas_cumprod[0]
    sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
    c = torch.sqrt(1 - alpha_prev - sigma * sigma)

    x = torch.sqrt(alpha_prev) * pred_x0 + c * predicted_noise + sigma * torch.randn_like(x)
    return x

    """
    Compute DDIM reverse update from timestep t -> t_prev.
    """
    # current and next alpha cumprod
    alpha_t = alphas_cumprod[t]
    alpha_prev = alphas_cumprod[t_prev] if t_prev >= 0 else alphas_cumprod[0]

    # predict x0 (clean image)
    pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

    # compute ddim coefficients
    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
    c = torch.sqrt(1 - alpha_prev - sigma_t**2)

    # update sample
    x_prev = (
        torch.sqrt(alpha_prev) * pred_x0 +
        c * predicted_noise +
        sigma_t * torch.randn_like(x_t)
    )
    return x_prev


def denoise_ddpm(x, t, alphas, alphas_cumprod, betas, pred):

    if t > 0:
        noise = torch.randn_like(x)
    else:
        noise = torch.zeros_like(x) # ma ha maske??
        # noise = 0

    alpha = alphas[t]
    alpha_cumprod = alphas_cumprod[t]
    beta = betas[t]

    x = (1 / torch.sqrt(alpha)) * (
        x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred
    ) + torch.sqrt(beta) * noise

    return x


def get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda"):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    # sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return betas, alphas, alphas_cumprod


