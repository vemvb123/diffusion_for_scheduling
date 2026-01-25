"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""

import code.inference.check_inference_utils as schedule
from code.inference.check_inference_utils import check_when_inference_makes_final_schedule
import code.inference.guidence as guidence
import code.inference.inference as inference

import code.scheduling.schedule as schedule

import code.dataset_code.utils as dataset_code

import torch


import matplotlib

from code.scheduling.schedule import inferenced_schedule
matplotlib.use('Agg')




def get_inference_result(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):
    w = 60
    h = 10
    n_jobs = 10


    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_code.get_td_from_path(dataset_folder, instance_idx)

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

    target_assignments, proc_times, job_id, pos_job = dataset_code.get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")
    elapsed = None
    inference_assignments = None
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_id, pos_job, adj_model_path, n_samples)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :h, :w]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = schedule.show_order_clear(inference_assignments, w)
    else:
        inference_assignments = schedule.round_to_values(inference_assignments, w)

   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)






# HER
def compare_inference(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    w = 60
    h = 10
    n_jobs = 10

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_code.get_td_from_path(dataset_folder, instance_idx)

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



    target_assignments, proc_times, job_id, pos_job = dataset_code.get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")


    conditions = torch.cat([            
        proc_times,
        job_id,
        pos_job,
    ], dim=1)

    inference_assignments = None
    if model_type == "adj":
        print("running inference")
        # NORMAL INFERENCE
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm(proc_times, job_id, pos_job, adj_model_path, n_samples)
        print("running comparison inference")
        # INFERENCE DDIM
        comparison_inference_assignments, comparison_elapsed, comparison_assignments_over_time = inference.adj_inference_ddim(proc_times, job_id, pos_job, adj_model_path, n_samples)
        # INFERENCE GUIDE
        # guided_inference_assignments, guided_elapsed, guidence.guided_assignments_over_time, errors = guide_adj_inference(conditions,3+1, adj_model_path, n_samples, 1)

    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")
    

    print("skjekker når final result er...")
    print(f"comparison of assignlents.. {len(comparison_assignments_over_time)}")
    check_when_inference_makes_final_schedule(comparison_assignments_over_time, comparison_inference_assignments, order)
    check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)
  
    print(f"Inference took {elapsed} time")
    print(f"ddim inference took {comparison_elapsed} time")

    comparison_inference_assignments = comparison_inference_assignments[:, :, :h, :w]
    inference_assignments = inference_assignments[:, :, :h, :w]

     #for value in [x.item() for x in errors]:
    #    print(value)
 
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    #if order:
    inference_assignments = schedule.round_to_values(inference_assignments, 16)
    comparison_inference_assignments = schedule.round_to_values(comparison_inference_assignments, 16)
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
    td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)
    td_scheduled_comp = inferenced_schedule(comparison_inference_assignments, order, env, td.copy(), graph_save_path_comp, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    makespan_comp = td_scheduled_comp['time']
    print(makespan)
    print(makespan_comp)


print("ran")



model_type = "adj"
order = False

adj_model_path = None
enc_model_path = None
if order:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_2.pth'
else:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_1.pth'

dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'
instance_idx = 10

print("inference result")
# get_inference_result(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)
compare_inference(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)
exit()