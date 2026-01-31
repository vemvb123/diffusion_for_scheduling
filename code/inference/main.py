"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""
import code.inference.utils as utils
import code.inference.utils as schedule_utils
import code.inference.guidence as guidence
import code.inference.inference as inference
import code.inference.cache_inference as cache_inference
import code.scheduling.schedule as schedule
import code.dataset_code.utils as dataset_utils

import torch


import matplotlib

matplotlib.use('Agg')




def get_inference_result_cached(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs):

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_utils.get_td_from_path(dataset_folder, instance_idx)

    ## Lag datasett for brandimarte instanse mk01
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
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

    target_assignments, proc_times, job_ops_adj, ops_ma_adj = dataset_utils.get_feature_adj_from_instance(td, env, order)

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
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = cache_inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, h, w)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :h, :w]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = utils.show_order_clear(inference_assignments, w, ops_ma_adj)
    else:
        inference_assignments = utils.round_to_values(inference_assignments, w, ops_ma_adj)

   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)






def get_inference_result_mk10(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs):

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
    elapsed = None
    inference_assignments = None
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :h, :w]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = utils.show_order_clear(inference_assignments, w, ops_ma_adj)
    else:
        inference_assignments = utils.round_to_values(inference_assignments, w, ops_ma_adj)

   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)

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


def get_inference_result_444(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs):

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

    h_when_masked = 24
    w_when_masked = 24

    target_assignments, proc_times, job_ops_adj, ops_ma_adj = dataset_utils.get_feature_adj_from_instance(td, env, order, h_when_masked, w_when_masked)

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
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, h_when_masked, w_when_masked)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    

    inference_assignments = inference_assignments[:, :, :h, :w]
    ops_ma_adj = ops_ma_adj[:, :, :h, :w]
    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = utils.show_order_clear(inference_assignments, w, ops_ma_adj)
    else:
        inference_assignments = utils.round_to_values(inference_assignments, w, ops_ma_adj)

    print("Produced inference assignments:")
    print(inference_assignments)

   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order, ops_ma_adj)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/444"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    td_scheduled = schedule.inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)









print("ran")



model_type = "adj"
order = True

enc_model_path = None
adj_model_path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/444/adj_type_adj_order_{order}.pth'


dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
# dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01_TEST'
instance_idx = 10

print("inference result")
w = 16
h = 4
n_jobs = 4
get_inference_result_444(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs)
# compare_inference(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)

# w = 64
# h = 24
# n_jobs = 10

# get_inference_result_cached(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx, w, h, n_jobs)
# exit()