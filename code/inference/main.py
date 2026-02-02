"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""
import code.inference.utils as utils
import code.scheduling.utils as schedule_utils
import code.inference.guidence as guidence
import code.inference.inference as inference
import code.inference.cache_inference as cache_inference
import code.scheduling.schedule as schedule
import code.dataset_code.utils as dataset_utils

import torch


import matplotlib

matplotlib.use('Agg')

import torch


def zero_only_columns(x):
    # (if on CUDA) bring it to CPU for processing
    x_cpu = x.cpu()

    # remove leading batch dimension if present
    x2 = x_cpu.squeeze(0)  # now shape is [num_rows, num_columns]

    # find columns where all rows are zero
    zero_cols = (x2 == 0).all(dim=0).nonzero(as_tuple=True)[0].tolist()
    return zero_cols





def extract_env_actions(ma_seq_matrix: torch.Tensor,
                        ops_sequence_order: torch.Tensor,
                        n_jobs: int,
                        max_ops_per_job: int) -> torch.Tensor:

    """
    Reconstruct the RL4CO FJSP action sequence from a completed schedule matrix.

    Args:
        ma_seq_matrix: Tensor of shape (1,1,n_machines,n_cols) with schedule ranks.
        ops_sequence_order: Tensor of length n_cols, gives op index within job.
        n_jobs: number of jobs
        max_ops_per_job: maximum operations per job

    Returns:
        LongTensor: shape (total_scheduled_ops,) with action indices in env format.
    """
    # Squeeze out batch dims -> shape (n_machines, n_cols)
    ma = ma_seq_matrix.squeeze(0).squeeze(0)
    n_machines, n_cols = ma.shape

    # We'll collect (rank, job, op_idx, machine)
    schedule_entries = []

    for m in range(n_machines):
        for c in range(n_cols):
            rank = int(ma[m, c].item())
            if rank > 0:
                op_idx_in_job = int(ops_sequence_order[c].item())
                job_id = c // max_ops_per_job
                schedule_entries.append((rank, job_id, op_idx_in_job, m))

    # Sort entries by global schedule rank ascending,
    # and tie-break by machine index ascending (as per your mapping rule).
    schedule_entries.sort(key=lambda x: (x[0], x[3]))

    # Convert entries to RL4CO env action IDs
    # RL4CO env actions are flat IDs where:
    # action_id = machine * n_jobs + job_id
    action_seq = []
    for rank, job_id, op_idx_in_job, machine in schedule_entries:
        # Compute the flat action index
        action_id = machine * n_jobs + job_id
        action_seq.append(action_id)

    return torch.tensor(action_seq, dtype=torch.long)


def assert_sequence_respected(ma_seq_matrix, ops_sequence_order):
    """
    Raise RuntimeError if any real operation sequence is violated.

    Rules:
    - Ignore the LAST section (block) entirely.
    - For every other section:
        * Column ranks must strictly increase
        * No column may contain only zeros
    """
    print("in assert")
    print(ma_seq_matrix.shape)
    ma = ma_seq_matrix.squeeze(0).squeeze(0)  # (n_machines, n_columns)
    ops = ops_sequence_order.tolist()
    n = len(ops)

    # --- build blocks ---
    blocks = []
    i = 0
    while i < n:
        op = ops[i]
        start = i
        while i < n and ops[i] == op:
            i += 1
        end = i
        blocks.append((op, start, end))

    # --- ignore the last block ---
    blocks_to_check = blocks[:-1]

    # --- validate blocks ---
    for op, start, end in blocks_to_check:
        prev_rank = -1

        for col in range(start, end):
            col_vals = ma[:, col]
            nonzeros = col_vals[col_vals > 0]


            if nonzeros.numel() == 0:
                # unique non-zero values in the whole matrix
                unique_nonzero = torch.unique(ma[ma > 0])
                n_unique = unique_nonzero.numel()

                # check whether last section (last block in whole matrix) is all zeros
                last_start, last_end = blocks_to_check[-1][1], blocks_to_check[-1][2]
                last_section_all_zero = (ma[:, last_start:last_end] == 0).all().item()

                # check if any column has more than one unique non-zero value
                cols_with_multiple_values = []
                for col_idx in range(ma.shape[1]):
                    vals = torch.unique(ma[:, col_idx][ma[:, col_idx] > 0])
                    if vals.numel() > 1:
                        cols_with_multiple_values.append((col_idx, vals.tolist()))

                raise RuntimeError(
                    f"seq order: {ops_sequence_order}. "
                    f"Zero-only column {col} in operation block {op}. "
                    f"Unique non-zero values in ma_seq_matrix: {n_unique}. "
                    f"Last section all zeros: {last_section_all_zero}. "
                    f"Columns with multiple non-zero values (col_idx: values): {cols_with_multiple_values}"
                )


            # ❌ must strictly increase
            rank = int(nonzeros[0].item())
            if rank <= prev_rank:
                raise RuntimeError(
                    f"Sequence violation in block {op}: "
                    f"rank {rank} at col {col} <= previous {prev_rank}"
                )

            prev_rank = rank

    print("✔ Sequence OK — no violations!")





def check_if_respects_sequence(inferenced, max_n_ops):
    B, C, W, H = inferenced.shape
    assert C == 1, "Expected C=1 in the inferenced"

    for b in range(B):
        mat = inferenced[b, 0]  # shape: W x H

        # slide across columns in blocks of max_n_ops
        start = 0
        while start < H:
            # take up to max_n_ops columns (may be shorter at end)
            block = mat[:, start : min(start + max_n_ops, H)]

            single_vals = []
            for col_idx in range(block.shape[1]):
                col = block[:, col_idx]

                # find all nonzero values in the column
                nonzeros = col[col != 0]

                # if exactly 1 nonzero, record it; if 0 nonzeros, ignore
                if len(nonzeros) == 1:
                    single_vals.append(nonzeros.item())
                elif len(nonzeros) > 1:
                    # if multiple nonzero values, this is still an error
                    raise AssertionError(
                        f"Column {start + col_idx} in batch {b} "
                        f"has multiple nonzero values"
                    )

            # Now check strictly increasing sequence among recorded values
            for i in range(len(single_vals) - 1):
                if not (single_vals[i] < single_vals[i + 1]):
                    raise AssertionError(
                        f"Values not strictly increasing in columns "
                        f"{start}..{start + block.shape[1] - 1}: {single_vals}"
                    )

            start += max_n_ops

    print("All column groups passed the increasing test!")



def get_inference_result(problem_type, instance_idx, model_type, order: bool):
    # instantiate all return values as None
    (td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, valid_h, valid_w) = (None,) * 10

    if problem_type == "444":

        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
        env, td_ignore, generator_params = schedule.make_instance(
            n_ma=4, 
            n_jobs=4, 
            max_op_per_job=4, 
            min_op_per_job=4, 
            max_proc_time=50, 
            min_proc_time=5, 
            max_eligable_ma_per_op=4, 
            min_eligable_ma_per_op=4, 
            batch_size=1
        )
        mask_h = 24
        mask_w = 24
        valid_h = 4
        valid_w = 16
        adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/444/adj_type_adj_order_{order}.pth"

    elif problem_type == "mk01":

        filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
        parameters = dataset_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)

        env, td_ignore, generator_params = schedule.make_instance(
            n_ma=parameters['n_machines'], 
            n_jobs=parameters['n_jobs'], 
            max_op_per_job=parameters['most_operations'], 
            min_op_per_job=parameters['fewest_operations'], 
            max_proc_time=parameters['max_processing_time'], 
            min_proc_time=parameters['min_processing_time'], 
            max_eligable_ma_per_op=parameters['max_machine_options'], 
            min_eligable_ma_per_op=parameters['min_machine_options'], 
            batch_size=1
        )
        print(parameters)

        mask_h = 24
        mask_w = 64
        valid_h = 6
        valid_w = 60
        adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_type_adj_order_{order}.pth"



    td = dataset_utils.get_td_from_path(dataset_folder, instance_idx)
    target_assignments, proc_times, job_ops_adj, ops_ma_adj, ops_sequence_order, opt_actions = dataset_utils.get_feature_adj_from_instance(td, env, order, mask_h, mask_w, True)

    # TODO skjekk i morra for mk01
    """
    print(n_ops)
    levels = torch.tensor([(i + 1) / n_ops for i in range(n_ops)], device="cuda")
    unique_vals = torch.unique(target_assignments[:, :valid_h, :valid_w])              # get unique values
    sorted_desc = unique_vals.sort(descending=True) # sort high → low
    print("lik??")
    print(sorted_desc.values)  # tensor([7, 5, 3, 2, 1, 0]) 
    print(levels)
    print("exit")
    exit()
    """
    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    ops_ma_adj = ops_ma_adj.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")
    elapsed = None
    inference_assignments = None
    columns_done = None
    if model_type == "adj":
        after_ts_check = 900
        print("beginning on cache")
        threshold = 0.02
        n_ops = int(torch.count_nonzero(target_assignments))
        # inference_assignments, elapsed, assignments_over_time, columns_done, done_at_t = cache_inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, order, mask_h, mask_w, after_ts_check, valid_h, valid_w, n_ops, threshold )
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w)
        # print(done_at_t)
        print(columns_done)
        print("ran inference")
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :valid_h, :valid_w]
    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w]

    return td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w



def get_inference_result_cached(model_type: str, order: bool, instance_idx, w, h, n_jobs):
    print("inside get inference cached")
    td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w = get_inference_result("mk01", instance_idx, model_type, order)
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    print("herkafaen")
    print(inference_assignments.shape)
    print(ops_ma_adj.shape)
    print(target_assignments.shape)
    target_assignments = target_assignments[:, :, :valid_h, :valid_w]
    print(target_assignments.shape)
    print("result")
    print(inference_assignments)
    zero_cols = zero_only_columns(inference_assignments)
    print(f"zeo cols: {zero_cols}")
    n_ops = int(torch.count_nonzero(target_assignments))
    if order:
        inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)
    else:
        inference_assignments = utils.round_to_values(inference_assignments, w, ops_ma_adj)
    print(inference_assignments)
    # inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)
    assert_sequence_respected(inference_assignments, td["ops_sequence_order"])
   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESUeckT
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order, ops_ma_adj, valid_h, valid_w)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/mk01"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    n_machines = 6
    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs, n_machines, ops_sequence_order=td["ops_sequence_order"])
    



    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)


# HER
def compare_inference(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    w = 60
    h = 10
    n_jobs = 10

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_utils.get_td_from_path(dataset_folder, instance_idx)

    env, td_ignore, generator_params = schedule.make_instance(
        n_ma=4, 
        n_jobs=4, 
        max_op_per_job=4, 
        min_op_per_job=4, 
        max_proc_time=50, 
        min_proc_time=5, 
        max_eligable_ma_per_op=4, 
        min_eligable_ma_per_op=4, 
        batch_size=1
    )


    target_assignments, proc_times, job_ops_adj, ops_ma_adj = dataset_utils.get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    ops_ma_adj = ops_ma_adj.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")


    conditions = torch.cat([            
        proc_times,
        job_ops_adj,
        ops_ma_adj,
    ], dim=1)

    inference_assignments = None
    if model_type == "adj":
        print("running inference")
        # NORMAL INFERENCE
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
        print("running comparison inference")
        # INFERENCE DDIM
        comparison_inference_assignments, comparison_elapsed, comparison_assignments_over_time = inference.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
        # INFERENCE GUIDE
        # guided_inference_assignments, guided_elapsed, guidence.guided_assignments_over_time, errors = guide_adj_inference(conditions,3+1, adj_model_path, n_samples, 1)

    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")
    

    print("skjekker når final result er...")
    print(f"comparison of assignlents.. {len(comparison_assignments_over_time)}")
    utils.check_when_inference_makes_final_schedule(comparison_assignments_over_time, comparison_inference_assignments, order)
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)
  
    print(f"Inference took {elapsed} time")
    print(f"ddim inference took {comparison_elapsed} time")

    comparison_inference_assignments = comparison_inference_assignments[:, :, :h, :w]
    inference_assignments = inference_assignments[:, :, :h, :w]

     #for value in [x.item() for x in errors]:
    #    print(value)
 
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    #if order:
    inference_assignments = utils.round_to_values(inference_assignments, 16, ops_ma_adj)
    comparison_inference_assignments = utils.round_to_values(comparison_inference_assignments, 16, ops_ma_adj)
    #else:
    #    inference_assignments = show_order_clear(inference_assignments, 16)
    #    guided_inference_assignments = show_order_clear(guided_inference_assignments, 16)

    print("RESULT")
    print(inference_assignments)
    print(comparison_inference_assignments)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_name_comp = f"scheduled_model_type_{model_type} order_{order}_comp.png"
    graph_save_path = f"{graph_folder}/{graph_name}"
    graph_save_path_comp = f"{graph_folder}/{graph_name_comp}"

    # TODO fjern
    # td_scheduled = inferenced_schedule(target_assignments, order, env, td.copy(), graph_save_path)
    # TODO gjør seinere så ikke kommenter ut
    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)
    td_scheduled_comp = schedule.inferenced_schedule(comparison_inference_assignments, order, env, td.copy(), graph_save_path_comp, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    makespan_comp = td_scheduled_comp['time']
    print(makespan)
    print(makespan_comp)

   

print("ran")



model_type = "adj"
order = True

enc_model_path = None
adj_model_path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/444/adj_type_adj_order_{order}.pth'

dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
# dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01_TEST'
instance_idx = 10

print("inference result")
# w = 64
# h = 24
w = 16
h = 4
n_jobs = 4
# get_inference_result_444(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs)
# compare_inference(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)
# get_inference_result_mk01(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs)

# w = 64
# h = 24
# n_jobs = 10
model_type = "adj"
order = True
instance_i = 10
w = 16
h = 4
n_jobs = 4
get_inference_result_cached(model_type, order, instance_idx, w, h, n_jobs)
# exit()