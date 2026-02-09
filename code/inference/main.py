"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""
import code.inference.experimental
import code.inference.utils as utils
import code.scheduling.utils as schedule_utils
import code.inference.guidence as guidence
import code.inference.inference as inference
import code.inference.cache_inference as cache_inference
import code.scheduling.schedule as schedule
import code.dataset_code.utils as dataset_utils

import torch

import code.inference.experimental as experimental
import matplotlib

matplotlib.use('Agg')

import torch


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
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_200.pth"


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


    print("Running inference")
    elapsed = None
    inference_assignments = None
    columns_done = None
    if model_type == "adj":
        after_ts_check = 995
        print("beginning on cache")
        threshold = 0.02
        # SAMPLES
        n_samples = 32
        n_ops = int(torch.count_nonzero(target_assignments))

        # === DDIM
        #inference_assignments = experimental.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, sampling_steps = 50, ddim_eta = 0.0)
        #elapsed = None 
        #assignments_over_time = None

        ## == CACHE
        # inference_assignments, elapsed, assignments_over_time, columns_done, done_at_t = cache_inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, order, mask_h, mask_w, after_ts_check, valid_h, valid_w, n_ops, threshold )
        # ==== ERSTATTER BATCHES
        t_replace = 995
        # inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm_batch_influence(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, t_replace, ops_sequence_order, valid_h, valid_w, n_ops)
        # === VANLID
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, 200, True)
        eta = 0.9
        #inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, eta, ddim_steps)

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
    #zero_cols = zero_only_columns(inference_assignments)
    #print(f"zeo cols: {zero_cols}")
    n_ops = int(torch.count_nonzero(target_assignments))
    print("show")
    print(n_ops)
    print(inference_assignments.shape)
    if order:
        inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)
    else:
        inference_assignments = utils.round_to_values(inference_assignments, w, ops_ma_adj)
    print("the assignments")
    print(inference_assignments)
    # inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)
    
    dups = utils.count_duplicate_instances(inference_assignments)
    print(f"amount of same instances: {dups}")

    report, total_errors, error_list = utils.assert_sequence_respected(inference_assignments, td["ops_sequence_order"])
    for key, value in report.items():
        if value["total"] == 0:
            print(f"{key} .. {value} .. NO ERRORS")
        else:
            print(f"{key} .. {value}")
    print(f"Total errors: {total_errors}")
    # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESUeckT
    # utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order, ops_ma_adj, valid_h, valid_w)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/mk01"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    n_machines = 6
    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs, n_machines, error_list, ops_sequence_order=td["ops_sequence_order"])
    



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
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm_batch_influence(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
        print("running comparison inference")
        # INFERENCE DDIM
        comparison_inference_assignments, comparison_elapsed, comparison_assignments_over_time = code.inference.experimental.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
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