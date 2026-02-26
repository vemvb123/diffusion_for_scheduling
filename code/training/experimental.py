from code.training.noising import get_noised_x
from code.training.utils import mask_invalid

import code.inference.guide as guide
import torch
import torch.nn as nn


import logging
from diffusers import CosineDPMSolverMultistepScheduler


def run_epoch_cosine(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode, valid_h, valid_w):


    total_loss = 0.0

    for batch_idx, (target_assignments, proc_times, job_ops_adj, ops_ma_adj) in enumerate(loop):

        features = torch.cat([
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model_adj(model_input, t, type_t="timestep")

        pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)
        loss = nn.MSELoss()(pred_valid, noise_valid)

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)

















def run_epoch_guide(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode, valid_h, valid_w, n_ops, 
            scheduler=None):


    total_loss = 0.0

    for batch_idx, (target_assignments, proc_times, job_ops_adj, ops_ma_adj) in enumerate(loop):

        features = torch.cat([
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)

        noise, noised = None, None
        if scheduler == None:
            noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        else:
            noise = torch.randn_like(target_assignments)
            noised = scheduler.add_noise(target_assignments, noise, t)


        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()


        pred = model_adj(model_input, t, type_t="timestep")
        
        """
        # må få tak i x0, 
        x0 = scheduler.step(pred, t, noised).pred_original_sample
        # så skjekke om inneholder invalid, så få et guide loss fra det .. 
        # TODO fiks her
        guide_loss = guide.amt_errors(x0, n_ops, ops_ma_adj, ops_seq_order, valid_h, valid_w, max_allowed_total_errors=50)
        # deretter minimerer loss med loss scale .. forventer at når korrupt er høy. så er det greit med mange infeas, så guide burde være lav
        guide_loss = guide_loss * 0.1

        """
        pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)

        #loss = nn.MSELoss()(pred_valid, noise_valid) * guide_loss
        loss = nn.MSELoss()(pred_valid, noise_valid)


        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)

import code.inference.utils as inference_utils




def top_n_valid(x, valid, n):
    """
    x:     (B, 1, 60, 6)
    valid: (B, 1, 60, 6) with 0/1
    n:     number of values to keep per batch instance
    """

    B = x.shape[0]

    # Flatten spatial dimensions
    x_flat = x.view(B, -1)            # (B, 360)
    valid_flat = valid.view(B, -1)    # (B, 360)

    # Mask invalid values
    x_masked = x_flat.masked_fill(valid_flat == 0, float('-inf'))

    # Top-n per batch
    top_vals, top_idx = torch.topk(x_masked, n, dim=1)

    # Create output tensor
    out_flat = torch.zeros_like(x_flat)

    # Scatter top-n values back
    out_flat.scatter_(1, top_idx, top_vals)

    # Reshape back to original shape
    return out_flat.view_as(x)


def get_job_lengths(job_ops_adj):
    return None

def count_empty_columns(x):
    empty_cols = (x == 0).all(dim=(1, 2))   # shape: (B, W)
    count = empty_cols.sum(dim=1)           # shape: (B,)
    return count



def count_group_violations(col_max_b, empty_col_b, group_sizes):
    """
    col_max_b:   (W,)
    empty_col_b: (W,)
    group_sizes: list[int]
    """
    count = 0
    start = 0

    for g in group_sizes:
        end = start + g
        vals = col_max_b[start:end]
        empty = empty_col_b[start:end]

        # Compare consecutive columns inside the group
        for i in range(len(vals) - 1):
            # Skip if next column is empty
            if empty[i + 1]:
                continue

            # Count non-increase
            if vals[i + 1] <= vals[i]:
                count += 1

        start = end

    return count

def count_batch_violations_predecessor(x, group_lists):
    """
    x: (B, 1, H, W)
    group_lists: list of lists, length B
    """
    col_max = x.max(dim=2).values.squeeze(1)    # (B, W)
    empty_col = (x == 0).all(dim=(1, 2))        # (B, W)

    counts = []
    for b in range(x.shape[0]):
        c = count_group_violations(
            col_max[b],
            empty_col[b],
            group_lists[b]
        )
        counts.append(c)

    return torch.tensor(counts)


# x (B, C, H, W)
def calculate_penalty_count(x0, ops_ma_adj, job_ops_adj, valid_h, valid_w):
    x0 = x0[:, :, :valid_h, :valid_w]

    print("her...")
    print(job_ops_adj.shape)
    print(job_ops_adj)
    exit()

    job_lengths = get_job_lengths(job_ops_adj) # TODO fyll inn metode
    x0 = top_n_valid(x0, ops_ma_adj, n=1)

    empty_column_count = count_empty_columns(x0)
    predecessor_break_count = count_batch_violations_predecessor(x0, job_lengths)

    total_error = empty_column_count + predecessor_break_count
    return total_error




def run_epoch_penalty_feasibility(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode, valid_h, valid_w, 
            scheduler=None):


    total_loss = 0.0

    for batch_idx, (target_assignments, proc_times, job_ops_adj, ops_ma_adj) in enumerate(loop):

        features = torch.cat([
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue

        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)

        noise, noised = None, None
        if scheduler == None:
            noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        else:
            noise = torch.randn_like(target_assignments)
            noised = scheduler.add_noise(target_assignments, noise, t)


        model_input = torch.cat([
            noised,
            features,
        ], dim=1)

        if mode == "train":
            optimizer.zero_grad()

        pred = model_adj(model_input, t, type_t="timestep")

        print("checking")
        print(pred)
        print(t)
        print(noised)
        x0 = scheduler.step(pred, t, noised).pred_original_sample
        penalty_count = calculate_penalty_count(x0, ops_ma_adj, job_ops_adj, valid_h, valid_w)

        penalty_loss = None
        if penalty_count > x0.shape(0) * 2: penalty_loss = 0.2
        else: penalty_loss = 0.2 * (penalty_count / (x0.shape(0) * 2))

        pred_valid, noise_valid = mask_invalid(valid_h, valid_w, pred, noise, ops_ma_adj)

        loss = nn.MSELoss()(pred_valid, noise_valid) + penalty_loss

        if mode == "train":
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, total_loss / len(loop)





def run_epoch_feature(loop, device, timesteps,
            model_adj, model_enc, optimizer,
            batch_size, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, mode):

    total_loss = 0.0
    for batch_idx, (target_assignments, proc_times, job_id, pos_job) in enumerate(loop):
        """ 
        logging.info("dataset shapes")
        logging.info(proc_times.shape)
        logging.info(job_id.shape)
        logging.info(pos_job.shape)
        """

        features = torch.cat([
            proc_times,
            job_id,
            pos_job,
        ], dim=1)


        # logging.info(f"f shape {features.shape}")
        features = features.to(device, dtype=torch.float32)
        target_assignments = target_assignments.to(device, dtype=torch.float32)

        B = target_assignments.shape[0]
        if B != batch_size:
            logging.warning(f"Skipping batch {batch_idx} with size {B}")
            continue


        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        # logging.info(f't shape {t.shape}')

        # encode images
        optimizer.zero_grad()
        enc_f = model_enc(features, t, type_t="timestep")
        # logging.info(f"encoded f {enc_f.shape}")

        # Sample noise
        noise, noised = get_noised_x(t, target_assignments, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        #logging.info("catination")
        #logging.info(noise.shape)
        #logging.info(enc_f.shape)

        model_input = torch.cat([
                noised,
                enc_f,
            ], dim=1)

        #logging.info(model_input.shape)

        # Predict noise
        pred = model_adj(model_input, t, type_t="timestep")
        #logging.info(f"made prediction, {pred.shape}")

        pred_valid, noise_valid = mask_invalid(4, 16, pred, noise)
        loss = nn.MSELoss()(pred_valid,noise_valid)
        # loss = nn.MSELoss()(pred, noise)
        if mode == "train":
            loss.backward()
            optimizer.step()

        # Save the loss value
        loss_value = loss.item()
        total_loss += loss_value

        loop.set_postfix(loss=loss_value)

    return loop, model_adj, model_enc, total_loss / len(loop)