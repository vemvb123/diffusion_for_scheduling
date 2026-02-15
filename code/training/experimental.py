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