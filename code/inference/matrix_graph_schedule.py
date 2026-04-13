import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import code.inference.inferenced_to_schedule
import code.inference.data_inform as utils
from code.inference.utils import insert_column_gaps
from code.inference.top_values import topk_binary_matrix


def show_a_sched(schedule, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=0.5, show_values=False, i=None, file_name=None, use_colors=True):
    H, W = schedule.shape

    schedule = insert_column_gaps(schedule, job_lenghts, gap_size=1)
    overlay_matrix_gapped = insert_column_gaps(overlay_matrix, job_lenghts, gap_size=1)
    filled_mask_gapped = insert_column_gaps(filled_mask, job_lenghts, gap_size=1)

    H2, W2 = schedule.shape

    fig, ax = plt.subplots(figsize=(W2 * cell_size, H * cell_size))

    if use_colors:
        ax.matshow(schedule, cmap="gray_r")
        ax.matshow(overlay_matrix_gapped, cmap="Reds", alpha=0.6)

    if use_colors:
        yellow_mask = filled_mask_gapped == 1
        yellow_overlay = np.ma.masked_where(~yellow_mask, filled_mask_gapped)
        ax.matshow(yellow_overlay, cmap="Wistia", alpha=0.9)

    # grid
    ax.set_xticks(np.arange(-0.5, W2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    if show_values:
        for y in range(H):
            for x in range(W2):
                value = schedule[y, x]
                if not np.isnan(value):
                    ax.text(x, y, f"{value:.2f}", va="center", ha="center", fontsize=10)

    if i != None:
        path = f"{save_path}/{i+1}.png"
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(path)
    else:
        if file_name is None:
            file_name = "showcase"
        path = f"{save_path}/{file_name}.png"
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(path)


'''
def show_single_sched(x, index, ops_sequence_order, ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.5, show_values=False):
    job_lenghts = utils.get_job_lengths(ops_sequence_order)
    assignments_top_k_binary = topk_binary_matrix(x[index][:, :valid_h, :valid_w], n_ops, ops_ma_adj)
    overlay_matrix = ops_ma_adj[0, 0].cpu().numpy()
    filled_mask = assignments_top_k_binary[0].cpu().numpy()  # (H, W)
    x = x[index, 0].cpu().numpy()
    show_a_sched(x, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=cell_size, show_values=show_values, i=None)
'''

def show_single_sched(x, index, ops_sequence_order, ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.5, show_values=False,
                 proc_times=None, job_ops_adj=None, show_features=False):
    job_lenghts = utils.get_job_lengths(ops_sequence_order)
    assignments_top_k_binary = topk_binary_matrix(x[index][:, :valid_h, :valid_w], n_ops, ops_ma_adj)
    overlay_matrix = ops_ma_adj[0, 0].cpu().numpy()
    filled_mask = assignments_top_k_binary[0].cpu().numpy()  # (H, W)
    x = x[index, 0].cpu().numpy()



    show_a_sched(x, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=cell_size, show_values=show_values, i=None)

    # Showing features
    if show_features:
        ops_ma_adj2 = ops_ma_adj.squeeze()
        show_a_sched(ops_ma_adj2, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=cell_size, show_values=show_values, i=None, file_name="ops_ma_adj")
    if proc_times is not None:
        proc_times = proc_times.squeeze()
        show_a_sched(proc_times, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=cell_size, show_values=show_values, i=None, file_name="proc_times")
    if job_ops_adj is not None:
        job_ops_adj = job_ops_adj.squeeze()
        show_a_sched(job_ops_adj, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size=cell_size, show_values=show_values, i=None, file_name="job_ops_adj", use_colors=False)
    






def show_schedule_over_time(
    schedule_over_time: torch.Tensor,  # shape (batch, 1, H, W)
    ops_ma_adj: torch.Tensor,             # shape (1, 1, H, W), binary 0/1
    save_path: str,
    given_assignments: torch.Tensor,
    job_lenghts: list[int],
    show_values: bool = False,
    cell_size: float = 0.5,   # NEW: controls how large each cell is
):

    os.makedirs(save_path, exist_ok=True)

    overlay_matrix = ops_ma_adj[0, 0].cpu().numpy()
    filled_mask = given_assignments[0].cpu().numpy()  # (H, W)

    for i in range(schedule_over_time.shape[0]):
        schedule = schedule_over_time[i, 0].cpu().numpy()
        show_a_sched(schedule, job_lenghts, overlay_matrix, filled_mask, save_path, cell_size, show_values, i)


def show_sched(index_to_check, ops_ma_adj, inference_assignments, assignments_over_time, ops_sequence_order, n_ops, valid_h, valid_w, save_path, over_time=False):
    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w]
    given_assignments = code.inference.top_values.topk_binary_matrix(inference_assignments[index_to_check][:, :valid_h, :valid_w], n_ops, ops_ma_adj)
    # given_assignments = code.inference.top_values.max_per_column_binary_matrix(inference_assignments[index_to_check][:, :valid_h, :valid_w], ops_ma_adj)
    result = torch.stack([t[index_to_check][:, :valid_h, :valid_w] for t in assignments_over_time])

    ops_ma_adj = ops_ma_adj.to("cuda")
    result = torch.cat([result, ops_ma_adj], dim=0)

    values = np.linspace(0, 1, n_ops+1)[1:]  # remove the 0
    print(values)
    job_lenghts = utils.get_job_lengths(ops_sequence_order)
    code.inference.matrix_graph_schedule.show_schedule_over_time(
        result,
        ops_ma_adj,
        save_path,
        given_assignments,
        job_lenghts,
        show_values = True,
        cell_size = 1
    )