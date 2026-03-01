# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
from code.inference.denoise import denoise_ddim, denoise_ddpm, get_inference_schedule
from diffusers import DDIMScheduler,  DDPMScheduler
#from torchdiff.ddim import VarianceSchedulerDDIM, ForwardDDIM, ReverseDDIM, SampleDDIM
from denoising_diffusion_pytorch import GaussianDiffusion



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




def adj_inference_ddpm_batch_improvement(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked, valid_h=None, valid_w=None, timesteps=1000, cos=False, t_replace=None, ops_sequence_order=None, n_ops=None):
    
    device = "cuda"
    print(f"Using model {model_path}, with timesteps {timesteps}, and cos: {cos}")

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
    
    x = None
    with torch.no_grad():
        
        x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)

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

            if cos:
                x = scheduler.step(predicted_noise, t, x).prev_sample
            else:
                x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)


            if t_replace != None and ( t_replace == timesteps - t ):
                print("in t replace")
                x_with_order = utils.show_order_clear(x, n_ops, ops_ma_adj, r_global=True)
                report, total_errors, error_list = utils.count_infeasibilities(x_with_order, ops_sequence_order, False, valid_h, valid_w)
                #print(f"replacing at timestep {t}, which forward in time is {timesteps - t}")
                #print(f"error list was: {error_list}")
                x = utils.replace_batches_with_fittest(x, error_list)

                #x_with_order = utils.show_order_clear(x, n_ops, ops_ma_adj, r_global=True)
                #report, total_errors, error_list = utils.count_infeasibilities(x_with_order, ops_sequence_order, False, valid_h, valid_w)
                #print(f"error list was after replacement: {error_list}")







            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments




def inference_lookahead(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked, timesteps=1000, jump=None, cos=False, smart_init=False):
    
    device = "cuda"
    print(f"Using model {model_path}, with timesteps {timesteps}, and cos: {cos}")

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
    variation_over_time = []
    
    x = None
    with torch.no_grad():
        if smart_init == False:
            x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)
        # ===
        elif smart_init == True:

            valid_h = 6   # valid rows
            valid_w = 55  # valid columns
            device = 'cuda'  # or 'cpu'

            # initialize tensor with zeros (full shape including padding)
            x = torch.zeros(n_samples, 1, h_when_masked, w_when_masked, device=device)

            # --- generate random row indices per sample and per valid column ---
            row_idx = torch.randint(0, valid_h, size=(n_samples, valid_w), device=device)

            # --- generate random normalized values per sample and per valid column ---
            values = torch.randint(1, 56, size=(n_samples, valid_w), device=device, dtype=torch.float32) / 55.0

            # --- column indices (valid columns only) ---
            col_idx = torch.arange(valid_w, device=device)

            # --- assign the values ---
            # x shape: [batch, 1, h, w], we assign using advanced indexing
            # we need to expand col_idx to match batch
            for b in range(n_samples):
                x[b, 0, row_idx[b], col_idx] = values[b]

        # ===

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
            if jump is not None and t == timesteps:
                t = jump
            t_tensor = torch.ones(n_samples, device=device).long() * t

            inputs = torch.cat([
                x,
                features,
            ], dim=1)


            predicted_noise = model(inputs, t_tensor, type_t="timestep")
            print(t)
            if t == 2:
                # TODO gjør flere prediksjoner
                for i in range( 2 ):
                    pred = model(inputs, t_tensor, type_t="timestep")
                    predicted_noise = torch.cat((predicted_noise, pred), dim=0)
                
                # TODO for alle preduksjoenen, trår aller mest fram .. disse er kopier av instansene, endrer ikke de predikerte
                x0_all = scheduler.step(predicted_noise, t, x.repeat(3,1,1,1)).prev_sample
                # TODO metode som rangerer instansene ut i fra eks mengde feil, antydd makepan..
                counts = ranger_etter_feil(x0_all, 6, 55, ops_ma_adj)
                # TODO kartlegger de 32 beste til de predikerte, beholder kun de predikerte med potensiale
                values, indices = torch.topk(counts, k=32, largest=False, sorted=True)
                # TODO beholder kun de 32 beste instansene 
                predicted_noise = predicted_noise[indices]


            if cos:
                x_t = scheduler.step(predicted_noise, t, x).prev_sample
                diff = x_t[:, :, :6, :60] - x[:, :, :6, :60]
                if t == 0 or t == 10 or t == 25 or t == 50 or t == 75 or t ==t ==  99:
                    variation = diff.abs().sum(dim=-1).squeeze(1)
                    variation_over_time.append(variation)
                x = x_t
            else:
                x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

            if t < 20:
                given_assignments.append(x.clone())

            """
            if t % 100==0:
                given_assignments.append(x.clone())
            """

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments, variation_over_time



def round_to_values(x, ops_ma_adj):
    n = 55

    V = ops_ma_adj.expand_as(x)
    x_masked = x.masked_fill(V == 0, float('-inf'))
    x_flat = x_masked.view(x.size(0), -1)  # (32, 360)
    _, topk_idx = torch.topk(x_flat, n, dim=1)
    out_flat = torch.zeros_like(x_flat)
    out_flat.scatter_(1, topk_idx, 1.0)
    out = out_flat.view_as(x)

    return out



def ranger_etter_feil(x, valid_h, valid_w, ops_ma_adj):
    # TODO metode som rangerer instansene ut i fra eks mengde feil, antydd makepan..
    x = x[:, :, :valid_h, :valid_w].to("cuda")
    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w].to("cuda")

    x = round_to_values(x, ops_ma_adj)

    col_counts = x.sum(dim=2).squeeze(1)  # (32, 6)
    extra = torch.clamp(col_counts - 1, min=0)  # (32, 6)
    extra_per_sample = extra.sum(dim=1)  # (32,)
    result_list = extra_per_sample.tolist()

    return torch.tensor(result_list)






def adj_inference_ddim(
    proc_times, job_ops_adj, ops_ma_adj,
    model_path, n_samples,
    h_when_masked, w_when_masked,
    sampling_steps = 100,   # fewer steps than 1000
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