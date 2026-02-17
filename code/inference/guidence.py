
import code.inference.guide as guide
import deepinv
from pathlib import Path
import logging

from code.inference.guide import guiding_function
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)
from diffusers import CosineDPMSolverMultistepScheduler, DDPMScheduler


import time
from pathlib import Path
import torch

from code.inference.denoise import get_inference_schedule



def adj_inference_ddpm_cos(job_lengts, proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked, valid_h, valid_w, n_ops, ops_seq_order, timesteps=1000, guidence_scale=0.5):
    
    device = "cuda"
    print(f"Using model {model_path}, with timesteps {timesteps}")

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

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

    given_assignments = []
    guidance_loss_scale = 5 # Adjust this value, between 5 and 100

    # Don't use torch.no_grad() for the entire loop - we need gradients!
    x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)

    # Prepare features
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

        # Predict noise (no gradients needed for model forward pass)
        with torch.no_grad():
            predicted_noise = model(inputs, t_tensor, type_t="timestep")

        # --- Guidance step (following tutorial pattern) ---
        # 1. Detach and enable gradients for x
        x = x.detach().requires_grad_()
        
        # 2. Get the predicted x0 from current x and noise prediction
        x0 = scheduler.step(predicted_noise, t, x).pred_original_sample

        
        # 3. Calculate guidance loss based on x0
        print("into loss")
        #loss = guide.amt_errors(x0, n_ops, ops_ma_adj, ops_seq_order, valid_h, valid_w, 80) * guidance_loss_scale
        #loss = guide.use_ma_less(x0, valid_h, valid_w) * guidance_loss_scale
        #loss = guide.minimize_infeasibility(x0, valid_h, valid_w, job_lengts) * guidance_loss_scale 
        # TODO mask vekk invalid allokasjoner
        loss = guide.increasing_columns_loss(x0, job_lengts, valid_h, valid_w) * guidance_loss_scale
        #loss = guide.loss_single_ma(x0, valid_h, valid_w)
        #loss = (loss_pred + loss_single_ma) * guidence_scale

        print(f"Step {t}, loss: {loss.item()}")
        
        # 4. Get gradient
        cond_grad = -torch.autograd.grad(loss, x)[0]
        
        # 5. If gradient exists, apply it
        if cond_grad is not None:
            x = x.detach() + cond_grad
        else:
            x = x.detach()
        # --- End guidance step ---
        
        # Now step with scheduler (using the guided x)
        x = scheduler.step(predicted_noise, t, x).prev_sample
        """
        if cos:
            x = scheduler.step(predicted_noise, t, x).prev_sample
        else:
            x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)
        """
        if t % 100 == 0:
            given_assignments.append(x.clone())


    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)
    given_assignments.append(x.clone())
    return x, elapsed, given_assignments



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



