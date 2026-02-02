import deepinv
from pathlib import Path
import torch
import time

import code.inference.denoise as denoise



def apply_confidence_mask(x, columns_done, columns_done_values, threshold, order, n_ops, denoised):
    if order:
        return apply_confidence_mask_order(x, columns_done, columns_done_values, threshold, n_ops, denoised)
    else:
        return apply_confidence_mask_no_order(x, columns_done, columns_done_values, threshold)




def apply_confidence_mask_order(
    x, columns_done, columns_done_values, threshold, n_ops, denoised
):
    """
    Commit confident columns using denoised values, only updating x for not-yet-locked columns.
    """
    B, C, W, H = x.shape
    device = x.device

    levels = torch.tensor([(i + 1) / n_ops for i in range(n_ops)], device=device)
    level_claimed = torch.zeros_like(levels, dtype=torch.bool)

    # columns not yet locked
    all_columns = torch.arange(H, device=device)
    columns_to_check = [c.item() for c in all_columns if c.item() not in columns_done]
    if len(columns_to_check) == 0:
        return x, columns_done, columns_done_values

    # extract candidate columns from denoised
    denoised_sub = denoised[:, 0, :, columns_to_check]  # B x W x n_cols

    for col_idx_local, col_global in enumerate(columns_to_check):
        vals = denoised_sub[..., col_idx_local]  # B x W

        # normalize column to [0,1] per batch
        col_min = vals.amin(dim=1, keepdim=True)
        col_max = vals.amax(dim=1, keepdim=True)
        norm = torch.where(
            col_max == col_min,
            torch.zeros_like(vals),
            (vals - col_min) / (col_max - col_min),
        )

        for b in range(B):
            if col_global in columns_done:
                continue

            diffs = torch.abs(levels.unsqueeze(0) - norm[b].unsqueeze(1))  # W x n_ops
            best_val_idx, best_level_idx = torch.unravel_index(torch.argmin(diffs), diffs.shape)
            best_dist = diffs[best_val_idx, best_level_idx]

            if best_dist <= threshold and not level_claimed[best_level_idx]:
                # lock column
                commit_val = levels[best_level_idx].item()
                columns_done.append(col_global)

                # commit in columns_done_values
                columns_done_values[b, 0, :, col_global] = 0.0
                columns_done_values[b, 0, best_val_idx, col_global] = commit_val

                level_claimed[best_level_idx] = True

    # finally, update x **only for columns not locked yet**
    unlocked_columns = [c for c in all_columns if c.item() not in columns_done]
    if len(unlocked_columns) > 0:
        x[:, 0, :, unlocked_columns] = denoised[:, 0, :, unlocked_columns]

    # ensure committed columns stay locked
    if len(columns_done) > 0:
        x[:, 0, :, columns_done] = columns_done_values[:, 0, :, columns_done]

    return x, columns_done, columns_done_values



def apply_confidence_mask_no_order(x, columns_done, columns_done_values, threshold):
    """
    Vectorized: commit confident columns across the whole batch at once.

    Args:
        x: tensor of shape (B, 1, W, H)
        columns_done: list of column indices already locked
        columns_done_values: tensor of same shape as x, storing committed values
        threshold: float, values >= threshold are committed

    Returns:
        x: tensor with done columns applied
        columns_done: updated list of locked columns
        columns_done_values: updated tensor of committed values
    """
    B, C, W, H = x.shape
    device = x.device

    all_columns = torch.arange(H, device=device)
    columns_to_check = [c.item() for c in all_columns if c.item() not in columns_done]

    if len(columns_to_check) == 0:
        return x, columns_done, columns_done_values

    # shape (B, W, n_cols_to_check)
    x_sub = x[:, 0, :, columns_to_check]

    # check which columns have any value >= threshold (shape: B x n_cols)
    eligible = (x_sub >= threshold).any(dim=1)  # B x n_cols

    # find which batch indices and column indices are eligible
    batch_idx, col_idx_in_sub = torch.nonzero(eligible, as_tuple=True)

    if len(batch_idx) == 0:
        # no columns to commit this step
        return x, columns_done, columns_done_values

    # map col_idx_in_sub to actual column indices
    col_idx = torch.tensor([columns_to_check[i.item()] for i in col_idx_in_sub], device=device)

    # get row index with max value in each eligible column per batch
    row_idx = torch.argmax(x[batch_idx, 0, :, col_idx], dim=1)

    # update committed values
    columns_done_values[batch_idx, 0, :, col_idx] = 0.0
    columns_done_values[batch_idx, 0, row_idx, col_idx] = 1.0

    # update done list
    for c in col_idx.tolist():
        if c not in columns_done:
            columns_done.append(c)

    # override x with committed values
    if len(columns_done) > 0:

        x[:, 0, :, columns_done] = columns_done_values[:, 0, :, columns_done]

    return x, columns_done, columns_done_values





# når får tilbake x, så minsker jeg det jeg får til kun x innenfor dimensjonene
def adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, model_path, n_samples, order: bool, h, w, after_ts_check, valid_h, valid_w, n_ops, threshold ):



    device = "cuda"

    model = deepinv.models.DiffUNet(
        in_channels=4, out_channels=1, pretrained=Path(model_path)
    ).to(device)

    # beta start var opprinnelig 1e-4
    beta_start = 1e-4
    beta_end = 0.02
    timesteps = 1000
    betas, alphas, alphas_cumprod = denoise.get_inference_schedule(beta_start, beta_end, timesteps, device = "cuda")

    model.eval()

    given_assignments = []
    
    x = None

    columns_done = []

    columns_done_values = torch.zeros_like(proc_times)  # same shape as x
    columns_done_values = columns_done_values.to("cuda")

    """
    for thing in [proc_times,, job_ops_adj, ops_ma_adj]:
        print(thing.shape)
    print("exit")
    exit()
    """
    done_at_t = None
    with torch.no_grad():
        
        x = torch.randn(n_samples, 1, h, w).to(device)

        features = torch.cat([            
            proc_times,
            job_ops_adj,
            ops_ma_adj,
        ], dim=1)

        features = features.to(device, dtype=torch.float32)
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

            denoised = None
            if after_ts_check <= (timesteps - t):
                denoised = denoise.denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)
            else:
                x = denoise.denoise_ddpm(x, t, alphas, alphas_cumprod, betas, predicted_noise)

            if after_ts_check <= (timesteps - t):
                x, columns_done, columns_done_values = apply_confidence_mask(x, columns_done, 
                                                                             columns_done_values, threshold, order, n_ops, denoised)

            required = set(range(n_ops))  # {0,1,...,15}
            if required.issubset(set(columns_done)) and done_at_t == None:
                done_at_t = timesteps - t

            if t % 100==0:
                given_assignments.append(x.clone())

    # end timer
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    x = torch.clamp(x, 0, 1)

    given_assignments.append(x.clone())
    return x, elapsed, given_assignments, columns_done, done_at_t


"""
for 0 og 1
se på schedule. hvis en verdi for en viss celle er en viss mengde mer enn de andre cellene, så sett verdien til 1.
Men vil ikke det ødelegge for schedulet? Det gjør kanskje mindre skade enn man skal tro.
Fordi resten av skedulat vil genereres med tanke på at den cellen er 1.
Man har en datastruktur som lagrer at den verdien er 1, og resten 0, så setter man de verdiene igjen i hver iteasjon.
Så bruker man det sammen med DDIM.
Man kan si at eks modellen går i 50 t, så begynner den å fastsette verdier. så går den eks for 200 t med ddim, eller inntill alle cellene er satt.
For varierende, så ser man ikke på alle celler, men kun de gyldiige cellene.

For order:
Her er det fortsatt kun 1 verdi blandt en kolonne som kan settes.
Problemet er fortsatt da å vite sekvensen.

Tenk verdiene denormalisert, så de varieer fra 1 - 16 / 1 - 60
Når en verdi er nærme nok en en verdi, eks 3 i rekkefølga, så blir den satt til den verdien.
men et problem er.. jeg tror diff verdiene er veldig lave før runding, så kan kanskje hende mange av dem er nærme de første verdiene,
og hva om flere av dem aldri blir nærme de seinere verdiene.... også å håndtere order er mer tricky, så kan kanskje få dårligere schedule, eller
infeasible.

litt vansklig. men tror ikke får til noe bedre, eller kommer ihvertfall ikke på noe bedre.



"""


