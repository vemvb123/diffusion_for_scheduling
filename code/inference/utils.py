from typing import List
import torch

import code.inference.utils as utils
from torch import Tensor


def fix_scheduled_multiple_times(x: torch.Tensor, valid_slots: torch.Tensor) -> torch.Tensor:
    print(x.shape)
    print(valid_slots.shape)
    B, _, R, C = x.shape

    # 1) Get all non-zero values per sample
    mask_nonzero = x != 0
    vals_per_batch = []
    for i in range(B):
        vals = x[i][mask_nonzero[i]]                   # select only non-zero
        vals_per_batch.append(vals)
        
    # 2) Build empty output same shape
    out = torch.zeros_like(x)
    for i in range(B):
        valid_pos = valid_slots[0].nonzero(as_tuple=True)

        vals = vals_per_batch[i]
        n_vals = min(len(vals), valid_pos[0].shape[0])

        if n_vals > 0:
            r_inds = valid_pos[1][:n_vals]
            c_inds = valid_pos[2][:n_vals]
            out[i, 0, r_inds, c_inds] = vals[:n_vals]


    return out

# takes a batch of solutions over time, specifically from a specific generated solution
# shape: B, C, H, W
# B is the amount of timesteps that has been recorded for the solution 
# take a specific solution of a batch of generated solutions, pair it with generated solutions for that specific solution over time
import matplotlib.pyplot as plt
import os
import numpy as np

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


import torch


def topk_binary_matrix(tensor: torch.Tensor, k: int, valid_slots: torch.Tensor):
    """
    tensor: shape (1, H, W)
    valid_slots: shape (1, 1, H, W) with 1s where allowed
    k: number of top values to select
    """
    # Make valid_slots the same shape as tensor
    mask = valid_slots[0, 0].bool()  # shape (H, W)
    
    # Flatten tensor and mask
    flat = tensor.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    # Only consider valid positions
    valid_values = flat.clone()
    valid_values[~mask_flat] = float('-inf')  # ignore invalid slots
    
    # Take top-k among valid positions
    topk_indices = torch.topk(valid_values, k).indices
    
    # Create binary result
    result = torch.zeros_like(flat)
    result[topk_indices] = 1
    
    return result.reshape_as(tensor)



def show_schedule_over_time(
    schedule_over_time: torch.Tensor,  # shape (batch, 1, H, W)
    overlay: torch.Tensor,             # shape (1, 1, H, W), binary 0/1
    save_path: str,
    filled_assignments: torch.Tensor,
    show_values: bool = False,
    cell_size: float = 0.5,   # NEW: controls how large each cell is
):  

    os.makedirs(save_path, exist_ok=True)

    overlay_matrix = overlay[0, 0].cpu().numpy()
    filled_mask = filled_assignments[0].cpu().numpy()  # (H, W)

    for i in range(schedule_over_time.shape[0]):
        schedule = schedule_over_time[i, 0].cpu().numpy()
        H, W = schedule.shape

        fig, ax = plt.subplots(figsize=(W * cell_size, H * cell_size))

        # base matrix
        ax.matshow(schedule, cmap="gray_r")

        # red overlay
        ax.matshow(overlay_matrix, cmap="Reds", alpha=0.6)

        # --- yellow overlay ---
        # intensity based on how close schedule value is to 1
        yellow_mask = filled_mask == 1
        yellow_overlay = np.ma.masked_where(~yellow_mask, filled_mask)

        ax.matshow(yellow_overlay, cmap="Wistia", alpha=0.9)


        # grid lines
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])

        if show_values:
            for y in range(H):
                for x in range(W):
                    value = schedule[y, x]
                    ax.text(
                        x, y,
                        f"{value:.2f}",
                        va="center",
                        ha="center",
                        fontsize=10
                    )

        plt.savefig(f"{save_path}/{i+1}.png", bbox_inches="tight", pad_inches=0)
        plt.close()





def make_job_lengths(ops_sequence_order: torch.Tensor) -> List[int]:
    flat = ops_sequence_order.flatten()
    vals = flat.tolist()

    job_lengths = []
    current_len = 0
    expected_next = 0

    for v in vals:
        if v == expected_next:
            current_len += 1
            expected_next += 1
        else:
            # run ended → store and start new run
            job_lengths.append(current_len)
            current_len = 1
            expected_next = v + 1

    # append the final run
    if current_len > 0:
        job_lengths.append(current_len)
    
    return job_lengths


def fix_breaks_predecessor(x: torch.Tensor, job_lengths: List) -> torch.Tensor:
    B, _, R, C = x.shape
    assert sum(job_lengths) == C, "job_lengths must sum to the number of columns"

    out = torch.zeros_like(x)

    # iterate over batch
    for b in range(B):
        col_start = 0
        for group_size in job_lengths:
            col_end = col_start + group_size

            # 1) Extract group columns
            group_vals = x[b, 0, :, col_start:col_end]           # shape (60, group_size)
            
            # 2) Get nonzero values -- should be exactly group_size ideally
            nonzeros = group_vals[group_vals != 0]               # shape (n_nonzero,)
            
            # 3) Sort them
            if len(nonzeros) > 0:
                sorted_vals, _ = torch.sort(nonzeros)            # ascending order
            else:
                sorted_vals = torch.tensor([], device=x.device)

            # 4) Place them back in increasing order
            #    We assume exactly 1 nonzero per column, so we fill from left to right
            #    leaving zeros if count < group_size
            for i, v in enumerate(sorted_vals[:group_size]):
                out[b, 0, :, col_start + i] = v

            col_start = col_end
    return out



def fix_infeasibilities(x: torch.Tensor, valid_slots: torch.Tensor, ops_sequence_order: torch.Tensor) -> torch.Tensor:
    print("in fix")
    x = fix_scheduled_multiple_times(x, valid_slots)
    #job_lengths = make_job_lengths(ops_sequence_order)  
    #x = fix_breaks_predecessor(x, valid_slots, job_lengths)
    return x










def round_to_values(
    x: torch.Tensor,
    n_values: int,

    valid_slots: torch.Tensor,
) -> torch.Tensor:
    """

    For each column:
    - If valid_slots has at least one valid entry, pick max among valid slots
    - If valid_slots is all zeros, ignore it and pick max am
ong all values
    """

    x = x.to("cuda")
    valid_slots = valid_slots.to("cuda")

    orig_shape = x.shape

    # Flatten leading dims -> [B, H, W]
    flat_x = x.view(-1, x.shape[-2], x.shape[-1])
    flat_vs = valid_slots.view(-1, x.shape[-2], x.shape[-1])

    # Check if there is ANY valid slot per (batch, column)
    # shape: [B, 1, W]
    has_any_valid = flat_vs.any(dim=1, keepdim=True)

    # Mask invalid slots with -inf
    neg_inf = torch.finfo(flat_x.dtype).min
    masked_x = torch.where(flat_vs.bool(), flat_x, neg_inf)

    # Max among valid slots
    max_valid, _ = masked_x.max(dim=1, keepdim=True)

    # Max among all slots (fallback)
    max_all, _ = flat_x.max(dim=1, keepdim=True)

    # Choose which max to use:
    # - use valid max if any valid exists
    # - otherwise use full max
    max_vals = torch.where(has_any_valid, max_valid, max_all)

    # Build result mask
    mask = torch.isclose(flat_x, max_vals)

    # If valid slots exist, enforce them
    mask = torch.where(has_any_valid, mask & flat_vs.bool(), mask)

    result = mask.float().view(orig_shape)
 
    return result


"""
# valid tensor: 1 for valide op-ma relasjoner, 0 for ikke.
# får datastructur fra "ops_ma_adj"
def round_to_values(x: Tensor, n_values: int, valid_slots: Tensor) -> torch.Tensor:
    orig_shape = x.shape

    # assume shape [1,1,4,16] or similar, so flatten leading dims
    flat = x.view(-1, x.shape[-2], x.shape[-1])  # [B, 4, 16]

    # find max in each column along row dim (dim=1)
    max_vals, _ = flat.max(dim=1, keepdim=True)  # [B, 1, 16]

    # compare with max and binarize
    mask = torch.isclose(flat, max_vals)  # True where value == max

    # convert to float (1.0/0.0)
    result = mask.float()

    # restore original leading dims
    result = result.view(orig_shape)

    return result
    # flatten and get top k indices
    topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

    # start with all zeros
    y = torch.zeros_like(x).flatten()

    # set top entries to 1
    y[topk_idx] = 1.0

    # reshape back to original shape
    return y.view(x.shape)

    """

def show_order_clear(x, n_values, valid_slots, r_global=True):
    """
    Rank the top-n_values entries.
    - If valid_slots has any 1s: only consider those positions
    - If valid_slots is all zeros: ignore valid_slots
    """
    if r_global:

        device = x.device
        x = x.to(device)
        valid_slots = valid_slots.to(device)
        valid_slots = valid_slots.repeat(x.shape[0], 1, 1, 1)

        B = x.shape[0]

        # Flatten per batch: (B, N)
        flat_x = x.reshape(B, -1)
        flat_vs = valid_slots.reshape(B, -1)
    

        # Check if each batch item has any valid slots
        has_any_valid = flat_vs.any(dim=1, keepdim=True)  # (B, 1)

        neg_inf = torch.finfo(flat_x.dtype).min

        # Mask invalid slots ONLY if that batch item has any valid slots
        masked_x = torch.where(
            has_any_valid,
            torch.where(flat_vs.bool(), flat_x, neg_inf),
            flat_x
        )

        # Top-k per batch
        topk_vals, topk_idx = torch.topk(masked_x, n_values, dim=1)

        # Sort top-k values so smallest rank = 1
        _, sorted_order = torch.sort(topk_vals, dim=1, descending=False)
        topk_idx_sorted = torch.gather(topk_idx, 1, sorted_order)

        # Output tensor
        out = torch.zeros(
            flat_x.shape,
            device=flat_x.device,
            dtype=torch.int64
        )
        ranks = torch.arange(
            1, n_values + 1,
            device=flat_x.device,
            dtype=torch.int64
        ).view(1, -1).expand(B, -1)

        out.scatter_(1, topk_idx_sorted, ranks)

        return out.reshape_as(x)


    else:

        device = x.device
        x = x.to(device)
        
        if valid_slots is not None:
            valid_slots = valid_slots.to(device)
        else:
            valid_slots = torch.ones_like(x)

        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        # Step 1: column-wise max (keeping only one value per column)
        col_max_mask = torch.zeros_like(x, dtype=torch.bool)
        for b in range(B):
            for c in range(C):
                for w in range(W):
                    col_vals = x[b, c, :, w]
                    col_vs = valid_slots[b, c, :, w]

                    # Mask if needed
                    if col_vs.any():
                        masked = torch.where(col_vs.bool(), col_vals, torch.finfo(x.dtype).min)
                    else:
                        masked = col_vals

                    # Index of maximum in this column
                    max_idx = torch.argmax(masked)
                    col_max_mask[b, c, max_idx, w] = True

        # Keep only column maxima
        filtered_x = torch.where(col_max_mask, x, torch.zeros_like(x))

        # Step 2: global ranking of surviving values
        flat_vals = filtered_x.flatten()
        nonzero_mask = flat_vals != 0

        out_flat = torch.zeros_like(flat_vals)
        if nonzero_mask.any():
            vals_to_rank = flat_vals[nonzero_mask]
            sorted_vals, order = torch.sort(vals_to_rank, descending=False)  # smallest -> 1
            rank_indices = nonzero_mask.nonzero(as_tuple=True)[0]

            # Assign ranks 1..n_ops (or less if fewer maxima)
            for rank, idx in enumerate(order, start=1):
                # Cap rank at n_ops
                out_flat[rank_indices[idx]] = min(rank, n_values)

        return out_flat.view_as(x)





# første skedulerte har minst verdi, sist skedulerte har størst verdi
def check_when_inference_makes_final_schedule(assignments_over_time: List[Tensor], final_assignment: Tensor, order: bool, valid_slots: Tensor, valid_h, valid_w):
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :valid_h, :valid_w]
        if order:
            assignment_at_time = utils.round_to_values(assignment_at_time, valid_w, valid_slots)
        else:
            assignment_at_time = utils.show_order_clear(assignment_at_time, valid_w, valid_slots)

        if torch.equal(assignment_at_time, final_assignment):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(final_assignment)
            break



def write_report(report, total_errors, error_list, print_report=False, save_as_file=None, n_ops=None, only_results=False):

    amt_feas = 0
    multi = 0
    seq = 0
    sum_multi_rate = 0.0
    sum_seq_rate = 0.0
    for key, value in report.items():

        if value["total"] == 0:
            amt_feas += 1
            case_str = f"{key} .. {value} .. NO ERRORS"
            if print_report: print(case_str)

        else:
            seq_rate = value["seq"] / n_ops if n_ops is not None else value["seq"]
            multi_rate = value["multi"] / n_ops if n_ops is not None else value["multi"]
            sum_multi_rate += multi_rate
            sum_seq_rate += seq_rate
            case_str = f"{key} .. {value} .. multi rate: {multi_rate:.2f}, seq rate: {seq_rate:.2f}"
            if value["multi"] > 0:
                multi += value["multi"]
            if value["seq"] > 0:                
                seq += value["seq"]

            if print_report: print(case_str)

    total_errors_str = f"Total errors: {total_errors} - scheduled multiple times: {multi}, break predecessor constraint: {seq}"
    avg_errors = f"Avg errors per instance: {total_errors / len(report):.2f}, scheduled multiple times: {multi / len(report):.2f}, break predecessor constraint: {seq / len(report):.2f}"


    avg_multi_rate = sum_multi_rate / len(report) 
    avg_seq_rate = sum_seq_rate / len(report) 
    avg_error_rates = f"Avg error rates: {avg_multi_rate + avg_seq_rate} - avg rate scheduled multi: {avg_multi_rate}, avg rate break predecessor constraint: {avg_seq_rate}"

    total_feas_str = f"Total feasible schedules: {amt_feas} / {len(report)}: {amt_feas / len(report) * 100:.2f}%"

    if only_results:
        return avg_multi_rate + avg_seq_rate

    if print_report:
        print(total_errors_str)
        print(avg_errors)
        print(avg_error_rates)
        print(total_feas_str)

    # write report to file
    if save_as_file is not None:
        with open(save_as_file, "w") as f:
            for key, value in report.items():
                if value["total"] == 0:
                    f.write(f"{key} .. {value} .. NO ERRORS\n")
                else:
                    f.write(f"{key} .. {value}\n")
            f.write(total_errors_str + "\n")
            f.write(avg_errors + "\n")
            f.write(avg_error_rates+ "\n")
            f.write(total_feas_str + "\n")






def add_makespans_report(min_makespan, max_makespan, avg_makespan, report_path):
    with open(report_path, "a") as file:
        file.write(f"makespans - avg {avg_makespan}, min {min_makespan}, max {max_makespan}")



def add_elapsed_time_report(elapsed, report_path):
    with open(report_path, "a") as file:
        file.write(f"Seconds to run inference: {elapsed} s")


def count_infeasibilities(ma_seq_matrix, ops_sequence_order, do_print=True, valid_h=None, valid_w=None, report_file_path=None, n_ops=None, only_results=False):
    if do_print != None:
        ma_seq_matrix = ma_seq_matrix[:, :, :valid_h, :valid_w]


    """
    Batch-wise validation of operation sequences.

    Returns:
        report: dict
            {run
                batch_idx: {
                    "zero": int,   # zero-only columns
                    "multi": int,  # multiple non-zeros in a column
                    "dup": int,    # duplicate values in a block
                    "seq": int,    # non-strictly-increasing sequences
                }
            }
    """
    total_feasible_schedules = 0

    # ---- normalize shape ----
    if ma_seq_matrix.dim() == 4:
        # (B, 1, M, C) -> (B, M, C)
        ma_seq_matrix = ma_seq_matrix.squeeze(1)

    B, n_machines, n_columns = ma_seq_matrix.shape
    error_list = [0] * B

    if do_print:
        print(f"B IS: {B}")

    ops = ops_sequence_order.tolist()
    n = len(ops)
    assert n == n_columns, "ops_sequence_order must match number of columns"

    # ---- build blocks (once) ----
    blocks = []
    start = 0
    for i in range(1, n):
        if ops[i] < ops[i - 1]:
            blocks.append((start, i))
            start = i
    blocks.append((start, n))

    blocks_to_check = blocks[:-1]  # ignore last block

    # ---- report ----
    report = {}

    # ---- iterate over batch ----
    total_errors = 0
    for b in range(B):
        ma = ma_seq_matrix[b]

        errors = {
            #"zero": 0,
            "multi": 0,
            #"dup": 0,
            "seq": 0,

            "total": 0
        }

        # ---- per-section processing ----
        for section_idx, (start, end) in enumerate(blocks_to_check):
            block_values = []

            for col in range(start, end):
                col_vals = ma[:, col]
                nonzeros = col_vals[col_vals > 0]

                if nonzeros.numel() == 0:
                    #errors["zero"] += 1
                    #errors["total"] += 1
                    #error_list[b] += 1
                    #total_errors += 1
                    block_values.append(None)

                elif nonzeros.numel() > 1:
                    errors["multi"] += 1
                    errors["total"] += 1
                    error_list[b] += 1
                    total_errors += 1
                    block_values.append("MULTI")

                else:
                    block_values.append(int(nonzeros.item()))

            if do_print:
                printable = [v if isinstance(v, int) else str(v) for v in block_values]
                print(f"B {b} Section {section_idx}: {printable}")

            # ---- section-level checks ----
            clean_vals = [v for v in block_values if isinstance(v, int)]
            """
            if len(clean_vals) != len(set(clean_vals)):
                errors["dup"] += 1
                errors["total"] += 1
                error_list[b] += n
                total_errors += 1
            """
            for i in range(1, len(clean_vals)):
                if clean_vals[i] <= clean_vals[i - 1]:
                    errors["seq"] += 1
                    errors["total"] += 1
                    error_list[b] += 1
                    total_errors += 1
                    break
        if do_print:
            print(" ")

        # ---- store only failing batches ----
        report[b] = errors

    if only_results:
        return write_report(report, total_errors, error_list, print_report=True, save_as_file=report_file_path, n_ops=n_ops, only_results=True)

    write_report(report, total_errors, error_list, print_report=True, save_as_file=report_file_path, n_ops=n_ops)

    return report, total_errors, error_list


# x: b, c, h, w
# errors: [total errors in batch 0, total errors in batch 1, etc...]

def replace_batches_with_fittest(x, errors):
    """
    Replace all batch instances with the best ones (least errors).
    
    Args:
        x: tensor of shape [B, C, H, W]
        errors: list or 1D tensor of length B, number of errors per batch
    Returns:
        x_new: tensor of shape [B, C, H, W] with “bad” batches replaced
    """
    device = x.device
    B = x.shape[0]

    # Convert errors to tensor if needed
    errors = torch.tensor(errors, device=device)

    # Step 1: find the minimum error
    min_err = torch.min(errors)

    # Step 2: indices of all batches with minimal error
    best_indices = (errors == min_err).nonzero(as_tuple=False).squeeze(-1)

    # Step 3: pick one of the best batches randomly
    chosen_idx = best_indices[torch.randint(len(best_indices), (1,), device=device)].item()

    # Step 4: replace all bad instances with the chosen best instance
    # mask for bad instances
    bad_mask = (errors != min_err)

    # create output
    x_new = x.clone()
    x_new[bad_mask] = x[chosen_idx]

    return x_new


def count_duplicate_instances(x: torch.Tensor) -> int:
    """
    x: Tensor of shape (B, C, H, W)
    Returns: number of batch instances that are exactly identical
             to at least one other instance.
    """
    B = x.shape[0]

    # Flatten each instance to (B, N)
    flat = x.view(B, -1)

    # Compare all pairs: (B, B, N)
    equal_matrix = flat.unsqueeze(1) == flat.unsqueeze(0)

    # Reduce over N → (B, B): True if entire instance matches
    same_instance = equal_matrix.all(dim=2)

    # Ignore self-comparisons (diagonal)
    same_instance.fill_diagonal_(False)

    # An instance is duplicate if it matches ANY other instance
    is_duplicate = same_instance.any(dim=1)

    # Count how many instances are duplicates
    return int(is_duplicate.sum().item())
