"""
inference.py contains code for running inference with trained models
"""
from code.inference.inferenced_to_schedule import show_order_clear
from code.inference.report_infeasibilities import count_infeasibilities
from diffusers import CosineDPMSolverMultistepScheduler, DDPMScheduler

from code.inference.denoise import denoise_ddpm, get_inference_schedule

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

def zero_unused_slots(x, ops_ma_adj, valid_h, valid_w, device):

    x = x.to(device, dtype=torch.float32)
    ops_ma_adj = ops_ma_adj.to(device, dtype=torch.float32)
    mask = ops_ma_adj.any(dim=2, keepdim=True)  # collapse 24
    x = x * mask


    return x




def adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, h_when_masked, w_when_masked, timesteps=1000, jump=None, cos=False, smart_init=False):
    
    device = "cuda"
    print(f"Using model {model_path}, with timesteps {timesteps}, and cos: {cos}")

    model = None
    optimizer = None

    try:
        model = deepinv.models.DiffUNet(
            in_channels=4,
            out_channels=1,
            pretrained=Path(model_path)
        ).to(device)

    except Exception as e:
        print('There was an error loading with path. Loading model by dictinary instead')
        # print(f'errir: {e}')
        checkpoint = torch.load(model_path, map_location=device)

        model = deepinv.models.DiffUNet(in_channels=4, out_channels=1, pretrained=None)
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # define optimizer BEFORE loading its state
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


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



    valid_h = 6   # valid rows
    valid_w = 55  # valid columns
    device = 'cuda'  # or 'cpu'



    given_assignments = []
    variation_over_time = []
    
    x = None
    with torch.no_grad():
        if smart_init == False:
            x = torch.randn(n_samples, 1, h_when_masked, w_when_masked).to(device)
        # ===
        elif smart_init == True:
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

            if cos:
                x_t = scheduler.step(predicted_noise, t, x).prev_sample
                diff = x_t[:, :, :6, :60] - x[:, :, :6, :60]
                if t == 0 or t == 10 or t == 25 or t == 50 or t == 75 or t ==t ==  99:
                    variation = diff.abs().sum(dim=-1).squeeze(1)
                    variation_over_time.append(variation)
                x = x_t
            else:
                x = denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

            # if t % 10 == 0:
            if t == 0:
                given_assignments.append(x.clone())

            """
            if t % 100==0:
                given_assignments.append(x.clone())
            """

            x = zero_unused_slots(x, ops_ma_adj, valid_h, valid_w, device)

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments, variation_over_time








def smoothmax(col, beta=20.0):
    # col shape [B,H]
    return torch.logsumexp(beta * col, dim=1) / beta


def only_increasing(x, valid_h, valid_w, job_lengths, eps=0.0):

    # remove dummy channel
    x = x[:, 0, :valid_h, :valid_w]   # [B,H,W]

    B, H, W = x.shape

    vals = []
    start = 0

    for length in job_lengths:
        end = start + length

        for j in range(start + 1, end):

            mj   = smoothmax(x[:, :, j])      # [B,H]
            mj_1 = smoothmax(x[:, :, j-1])    # [B,H]

            b = mj - mj_1 - eps
            vals.append(b)

        start = end

    if len(vals) == 0:
        return torch.empty(B, 0, device=x.device)

    return torch.stack(vals, dim=1)


'''
def solve_column_qp(u_nominal, x, valid_h, valid_w, job_lengths, eps=0.0):

    x = x.detach().clone().requires_grad_(True)
    u = u_nominal.clone()

    b = only_increasing(x, valid_h, valid_w, job_lengths, eps=eps)   # [B,K]

    # if already feasible
    if (b >= 0).all():
        return u

    # total = b.sum()
    mask = (b < 0).float()
    total = (mask * b).sum()
    grad_b = torch.autograd.grad(total, x)[0]

    # alpha like paper
    # alpha = b.clamp(min=0) + b
    alpha = -torch.relu(-b)   # only penalize violations
'''
    # alpha = torch.where(
    # b >= 0,
    # b,
    # 10.0 * b   # 🔥 stronger push when violated
    # )
'''
    # aggregate violation
    # lhs = (grad_b * u).sum(dim=(1,2), keepdim=True)
    lhs = (grad_b * u).sum(dim=(1,2,3), keepdim=True)

    # rhs = alpha.mean(dim=1, keepdim=True).unsqueeze(-1)
    rhs = alpha.mean(dim=1).view(-1,1,1,1)

    violation = lhs + rhs

    grad_norm = (grad_b * grad_b).sum(dim=(1,2), keepdim=True) + 1e-8

    lam = torch.clamp(-violation / grad_norm, min=0)

    # print("ham")
    # print(u.shape)
    # print(lam.shape)
    # print(grad_b.shape)

    u_safe = u + lam * grad_b

    return u_safe.detach()
'''

def solve_column_qp(u_nominal, x, valid_h, valid_w, job_lengths, t, eps=0.0, self_adjusting_gamma=False):

    x = x.detach().clone().requires_grad_(True)
    u = u_nominal.clone()

    b = only_increasing(x, valid_h, valid_w, job_lengths, eps=eps)   # [B,K]

    if (b >= 0).all():
        return u

    # 🔥 ONLY violated constraints
    mask = (b < 0).float()
    total = (mask * b).sum()

    grad_b = torch.autograd.grad(total, x)[0]

    # 🔥 simple and stable
    alpha = -torch.relu(-b)

    lhs = (grad_b * u).sum(dim=(1,2,3), keepdim=True)
    rhs = alpha.mean(dim=1).view(-1,1,1,1)

    violation = lhs + rhs

    grad_norm = (grad_b * grad_b).sum(dim=(1,2,3), keepdim=True) + 1e-8

    gamma = 1.0


    # gamma = 2.0 * torch.relu(-b).mean(dim=1).view(-1,1,1,1)
    # gamma = torch.relu(-b).mean(dim=1).view(-1,1,1,1)
    lam = torch.clamp((-violation + gamma) / grad_norm, min=0)

    u_safe = u + lam * grad_b

    return u_safe.detach(), gamma


def inference_guide(
        proc_times,
        job_ops_adj,
        ops_ma_adj,
        model_path,
        n_samples,
        h_when_masked,
        w_when_masked,
        timesteps=1000,
        jump=None,
        cos=False,
        smart_init=False,
        job_lengths=None,
        td=None,
        ):

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



    valid_h = 6   # valid rows
    valid_w = 55  # valid columns
    device = 'cuda'  # or 'cpu'



    given_assignments = []
    variation_over_time = []

    g_loss_over_time = []
    g_var = []

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
            x_denoised = scheduler.step(predicted_noise, t, x).prev_sample
            '''
            n_ops = sum(x for x in job_lengths if x != 1)
            ops_ma_adj_sc = ops_ma_adj[:, :, :valid_h, :valid_w].clone()

            report_file_path = f'results/report_g.txt'
            x_denoised_sc= x_denoised[:, :, :valid_h, :valid_w].clone()
            x_denoised_sc = show_order_clear(x_denoised_sc, n_ops,ops_ma_adj_sc, r_global=False) # tidligere order visning ... DENNE ER KLART BEDRE, far langt mindre feil for mk01
            report, total_errors_bg, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = count_infeasibilities(
            x_denoised_sc, td["ops_sequence_order"][:valid_w], report_file_path=report_file_path, n_ops=n_ops, do_print=False)
            '''
            # ---- 2. Convert to update (IMPORTANT) ----
            stop_guide = -2 # 6
            gamma = None
            if t > stop_guide:

                u_nominal = x_denoised - x   # diffusion direction
                # ---- 3. Solve QP (core of paper) ----

                with torch.enable_grad():
                    u_safe, gamma = solve_column_qp(
                        u_nominal,
                        x,
                        valid_h,
                        valid_w,
                        job_lengths,
                        t,
                        eps=0.0
                    )


                # ---- 4. Apply corrected update ----
                x = x + u_safe
            else:
                x = x_denoised
            '''
            report_file_path = f'results/report_g.txt'
            x_sc = x[:, :, :valid_h, :valid_w].clone()
            x_sc = show_order_clear(x_sc, n_ops, ops_ma_adj_sc, r_global=False) # tidligere order visning ... DENNE ER KLART BEDRE, far langt mindre feil for mk01
            report, total_errors_ag, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = count_infeasibilities(
                x_sc, td["ops_sequence_order"][:valid_w], report_file_path=report_file_path, n_ops=n_ops, do_print=False)
            
            if t == stop_guide:
                print(f'timestep reached {stop_guide}. Stopped guiding')
                print(f'infeasibilities before correction at timestep {t}: {total_errors_bg}, after correction: {total_errors_ag}')
            else:
                try:
                    print(f'infeasibilities before correction at timestep {t}, with gamma {gamma.mean().item()}: {total_errors_bg}, after correction: {total_errors_ag}')
                except:
                    print(f'infeasibilities before correction at timestep {t}, with gamma {gamma}: {total_errors_bg}, after correction: {total_errors_ag}')

            # ---- MEASURE VIOLATION ----
            b = only_increasing(x, valid_h, valid_w, job_lengths)
            loss = torch.relu(-b).sum(dim=1).mean().item()
            viol = (b < 0).sum(dim=1).float().mean().item()
            g_loss_over_time.append(loss)
            g_var.append(viol)

            '''
            # if t % 10 == 0:
            if t == 0:
                given_assignments.append(x.clone())

            # x = zero_unused_slots(x, ops_ma_adj, valid_h, valid_w, device)
    '''
    plt.plot(g_loss_over_time, label="Violation magnitude")
    plt.plot(g_var, label="Number of violations")

    plt.title("Constraint improvement over time")
    plt.xlabel("Diffusion step")
    plt.ylabel("Value")

    plt.legend()

    plt.savefig("results/g_loss.png")  # 🔥 saves image

    plt.close()  # optional but recommended
    '''

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments, variation_over_time

