# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
from code.inference.denoise import denoise_ddim, denoise_ddpm, get_inference_schedule


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


import deepinv
import torch


import time
from pathlib import Path

import code.inference.utils as utils


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



def adj_inference_ddpm_batch_improvement(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked,
                       t_replace=None, ops_sequence_order=None, valid_h=None, valid_w=None, n_ops=None):

    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Amount of trainable parameters: {trainable_params}")
    exit()


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

            if t_replace != None and ( t_replace == timesteps - t ):
                x_with_order = utils.show_order_clear(x, n_ops, ops_ma_adj, r_global=True)
                report, total_errors, error_list = utils.assert_sequence_respected(x_with_order, ops_sequence_order, False, valid_h, valid_w)
                print(f"replacing at timestep {t}, which forward in time is {timesteps - t}")
                print(f"error list was: {error_list}")
                x = utils.replace_batches_with_fittest(x, error_list)

                x_with_order = utils.show_order_clear(x, n_ops, ops_ma_adj, r_global=True)
                report, total_errors, error_list = utils.assert_sequence_respected(x_with_order, ops_sequence_order, False, valid_h, valid_w)
                print(f"error list was after replacement: {error_list}")


            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments


from diffusers import DDIMScheduler
#from torchdiff.ddim import VarianceSchedulerDDIM, ForwardDDIM, ReverseDDIM, SampleDDIM
from denoising_diffusion_pytorch import GaussianDiffusion

def adj_inference_ddim(
    proc_times, job_ops_adj, ops_ma_adj,
    model_path, n_samples,
    h_when_masked, w_when_masked,
    sampling_steps = 50,   # fewer steps than 1000
    ddim_eta = 0.0,         # eta=0 => deterministic DDIM
    timesteps=1000

):

    device = "cuda"


    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    model.eval()


    # 1) Build the DDIM scheduler
    beta_start = 1e-4
    beta_end = 0.02

    scheduler = DDIMScheduler(
        num_train_timesteps=timesteps,       # same as training
        beta_start=beta_start,                # same as training
        beta_end=beta_end,                    # same as training
        beta_schedule="squaredcos_cap_v2",    # cosine schedule
        clip_sample=True,
        prediction_type="epsilon",            # normally matches training
    )

    scheduler.set_timesteps(sampling_steps)

    # 3) Start from pure noise
    x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)

    features = torch.cat([
        proc_times,
        job_ops_adj,
        ops_ma_adj,
    ], dim=1)
  
    features = features.repeat(n_samples, 1, 1, 1)
    features = features.to(device, dtype=torch.float32)

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            t_tensor = torch.ones(n_samples, dtype=torch.long, device=device) * t

            inputs = torch.cat([x, features], dim=1)

            noise_pred = model(inputs, t_tensor, type_t="timestep")
            out = scheduler.step(noise_pred, t, x, eta=ddim_eta)
            x = out.prev_sample

    x = torch.clamp(x, 0, 1)
    return x, None, None