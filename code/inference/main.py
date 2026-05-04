
print("started running")
"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""
import code.dataset_code.benchmark_utils
import code.dataset_code.dataset_maker
import code.inference.experimental
import code.inference.infeasibilities
import code.inference.infeasibilities_try
import code.inference.inferenced_to_schedule
import code.inference.report_infeasibilities
import code.inference.utils as utils
import code.scheduling.utils as schedule_utils
import code.inference.guidence as guidence
import code.inference.inference as inference
import code.inference.cache_inference as cache_inference
import code.scheduling.schedule as schedule
import code.dataset_code.dataset_utils as dataset_utils
import code.inference.confidence_interval as confidence_interval_utils
import code.inference.matrix_graph_schedule as matrix_graph_schedule
import code.dataset_code.benchmark_utils as benchmark_utils

import torch
import code.inference.experimental as experimental
import matplotlib

matplotlib.use('Agg')

import torch
import numpy as np

def compute_job_lengths(indices):
    """
    Compute lengths of jobs from a 1D tensor of indices, where each
    job is defined as a contiguous sequence starting at 0 and increasing
    by +1. Trailing filler zeros are ignored.
    """
    arr = indices.tolist()
    job_lengths = []
    current_length = 0
    expected_next = 0

    for i, val in enumerate(arr):
        # If we see expected value in a sequence
        if val == expected_next:
            current_length += 1
            expected_next += 1

        # If we see a 0 where a new job could start
        elif val == 0:
            # If we already finished a valid job (current_length > 0),
            # we record it and start a new one
            if current_length > 0:
                job_lengths.append(current_length)
            current_length = 1
            expected_next = 1

        # Anything else breaks the job detection
        else:
            break

    # After loop, if we were in a valid job, save it
    if current_length > 0:
        job_lengths.append(current_length)

    return job_lengths


def get_problem_type(problem_type : str):
    (td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, valid_h, valid_w) = (None,) * 10

    if problem_type == "444":

        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
        env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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
        
        filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
        # TODO
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
        #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'
        parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)

        env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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
        valid_w = 55

        adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_type_adj_order_{order}.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_{timesteps}.pth"

    elif problem_type == "mk02":
        
        filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk02.txt'
        # TODO
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_10j_6ma_6op_mk02'
        #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'
        parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)

        env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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
        valid_w = 58

        adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_type_adj_order_{order}.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_{timesteps}.pth"

    elif problem_type == "mk03":
        
        filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk03.txt'
        # TODO
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_10op_mk03'
        #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'
        parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)

        env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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

        # TODO ikke sikker på eksakt størrelse.. modell funker uansett størrelse. men ytelsen er forferdelig..
        # veit ikke om ytedelse er forferdelig fordi ikke trent nok, eller feil mask_w
        # problem størrelsen er også en del større enn mk02, så kan være at bare problemet er så stort at det er veldig vansklig å ikke ha noe som helst feil
        # rate er 4.8 operasjoner, av 150.. så er ikke alt for ille
        mask_h = 24
        mask_w = 152
        valid_h = 8
        valid_w = 150

        adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk03/mk03_0.0001.pth'



    elif problem_type == "mk10":

        filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
        # TODO
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk10_20j_15ma_14op'
        #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'
        parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)

        env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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
        mask_w = 280
        valid_h = 15
        valid_w = 280

        adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk10/mk10.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_{timesteps}.pth"

    return dataset_folder, env, mask_h, mask_w, valid_h, valid_w, adj_model_path






def get_inference_result(problem_type, instance_idx, model_type, order: bool, benchmark_instance = None):
    print("starting inference")
    # instantiate all return values as None
    (td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, valid_h, valid_w) = (None,) * 10

    dataset_folder, env, mask_h, mask_w, valid_h, valid_w, adj_model_path = get_problem_type(problem_type)

    print(f"dataset_folder: {dataset_folder}")
    # TODO skjekk om samme dimensjoner
    # benchmark_instance = None
    if benchmark_instance is None:
        td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)
    else:
        td = benchmark_utils.make_td_from_benchmark_working(benchmark_instance)
    target_assignments, proc_times, job_ops_adj, ops_ma_adj, ops_sequence_order, opt_actions = dataset_utils.get_dataset_features(td, env, order, mask_h, mask_w, 
                                                                                                                                  include_ops_sequence=True, 
                                                                                                                                  from_benchmark_instance=benchmark_instance is not None,
                                                                                                                                  include_ops_actions=False)

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
    n_ops = None
    if model_type == "adj":
        after_ts_check = 995
        print("beginning on cache")
        threshold = 0.02
        # SAMPLES
        n_samples =32
        n_ops = int(torch.count_nonzero(target_assignments))

        timesteps = 300
        #sampling_steps = 60
        ddim_eta = 1.0
        
        graph_folder  = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/{problem_type}"
        graph_name = f"result_schedule_{problem_type}_bench.png"
        graph_save_path = f"{graph_folder}/{graph_name}"


        report_file_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_file_{timesteps}t_{n_samples}b_zero_{problem_type}.txt"
        report_file_path_fix = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_file_{timesteps}t_{n_samples}b_zero_{problem_type}.txt"
        # report_file_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_file_{timesteps}t_{n_samples}b.txt"
        # report_file_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_random_mk01.txt"

        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_{timesteps}.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_type_adj_order_{order}.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/cos_beta/timestep_1000_cos.pth"
        #adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/cos_beta/timestep_1000_beta.pth"
        ##adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/adj_timestep_200.pth"
        #adj_model_path = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/further_improved/trained_on_100_timesteps_sch_1000.pth"
        print(problem_type)
        adj_model_path = None
        if problem_type == 'mk10':
            adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk10/mk10.pth" # BRUK DENNE FOR MK10
        elif problem_type == 'mk01':
            adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/cos_beta/timestep_1000_cos.pth" # BRUK DENNE FOR MK01
        elif problem_type == 'mk02':
            adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk02/mk02.pth" # BRUK DENNE FOR MK02
        elif problem_type == 'mk03':
            adj_model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk03/mk03_0.0001.pth" # BRUK DENNE FOR MK03


        print(f"using model at path: {adj_model_path}")
        # === DDIM
        #inference_assignments, elapsed, assignments_over_time = experimental.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, sampling_steps = sampling_steps, ddim_eta = ddim_eta, timesteps = timesteps)
        #elapsed = None 
        #assignments_over_time = None
        ## == CACHE
        # inference_assignments, elapsed, assignments_over_time, columns_done, done_at_t = cache_inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, order, mask_h, mask_w, after_ts_check, valid_h, valid_w, n_ops, threshold )
        # ==== ERSTATTER BATCHES
        t_replace = 92
        cos = True
        #inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm_batch_influence(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, t_replace, ops_sequence_order, valid_h, valid_w, n_ops)
        #inference_assignments, elapsed, assignments_over_time = experimental.adj_inference_ddpm_batch_improvement(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, valid_h=valid_h, valid_w=valid_w, timesteps=timesteps, cos=True, t_replace=t_replace, ops_sequence_order=ops_sequence_order, n_ops=n_ops)
        # === VANLID

        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False)
        eta = 0.9

        '''
        # GUIDING
        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.inference_guide(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False,
            job_lengths=compute_job_lengths(td["ops_sequence_order"]),
            td=td
        )
        eta = 0.9
        '''
        """
        #inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, eta, ddim_steps)
        # === LOOK AHEAD
        """
        """
        inference_assignments, elapsed, assignments_over_time, variation_over_time = experimental.inference_lookahead(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False
            )
        """
        # ==== RANDOM
        """
        assignments_to_make = 20
        ops_seq_order = td["ops_sequence_order"]
        job_lengts = compute_job_lengths(ops_seq_order)
        print(f"job lengths: {job_lengts}")
        valid_w = 280
        inference_assignments = schedule.schedule_randomly(ops_ma_adj, valid_h, valid_w, job_lengts, N=n_ops)
        for i in range(assignments_to_make-1):
            instance_batch= schedule.schedule_randomly(ops_ma_adj, valid_h, valid_w, job_lengts)
            inference_assignments = torch.cat([inference_assignments, instance_batch], dim=0)


        print(f"random assignments shape: {inference_assignments.shape}")
        print(inference_assignments)
        elapsed = 0
        assignments_over_time = None
        """
        """
        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False)
        eta = 0.9
        """


        # === GUIDENCE
        #n_ops = int(torch.count_nonzero(target_assignments))
        ops_seq_order = td["ops_sequence_order"]
        job_lengts = compute_job_lengths(ops_seq_order)
        n_ops = sum(x for x in job_lengts if x != 1)

        #job_lengts = [5, 6, 5, 6, 6, 6, 5, 6, 6, 5, 0,0,0,0] 
        #inference_assignments, elapsed, assignments_over_time = guidence.adj_inference_ddpm_cos(job_lengts, proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, valid_h, valid_w, n_ops, ops_seq_order, timesteps, True)

        # === CHECK EVOLUTION
        #utils.show_sched(2, ops_ma_adj, inference_assignments, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, graph_save_path)




        # TODO skjekker i hvor stod grad skjeduelen over tid er lik resultatet
        # resultatet skal være slik: for hver instanse, en liste der det står hvor ofte hver siste verdi var verdiene i skederne over tid
        """
        inference_assignments = inference_assignments[:, :, :6, :60]
        ops_ma_adj = ops_ma_adj[:, :, :6, :60]
        inference_assignments = utils.show_order_clear(inference_assignments, n_ops, ops_ma_adj)
        scheds__order_clear = []
        for sched in assignments_over_time:
            sched = sched[:, :, :6, :60]
            sched_clear = utils.show_order_clear(sched, n_ops, ops_ma_adj)
            scheds__order_clear.append(sched_clear)

        print("orgi")
        first_batch = inference_assignments[0]           # selects batch index 0 (first batch)
        first_column = first_batch[:, 0]  # selects the first column
        print(first_column)

        print("dn")
        print(len(scheds__order_clear))

        ref = scheds__order_clear[0]
        all_same = all(torch.equal(ref, t) for t in scheds__order_clear)
        print("All tensors the same?", all_same)


        indexes_to_check = [20, 19, 18, 17]

        for i in indexes_to_check:
            print(f"t={i}")
            dn = scheds__order_clear[i-1]
            f_c = dn[0]           # selects batch index 0 (first batch)
            f_c_c = f_c[:, 0]  # selects the first column
            print(f_c_c)
            print("..")

        print("var over time")
        print(len(variation_over_time))
        print(variation_over_time[0].shape)
        for i in range(len(variation_over_time)):
            mean_x = variation_over_time[i].mean(dim=0, keepdim=True)  # shape: [1, 6]
            print(mean_x)


        print()
        print("exit")
        exit()
        """









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

    job_ops_adj = job_ops_adj[:, :, :valid_h, :valid_w]
    proc_times = proc_times[:, :, :valid_h, :valid_w]

    return td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w, report_file_path, adj_model_path, graph_save_path, report_file_path_fix, n_ops






def get_inference_result_cached(model_type: str, order: bool, instance_idx, w, h, n_jobs, problem_type, benchmark_instance = None):
    print("inside get inference cached")
    # TODO endre hvis bruker annet
    td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w, report_file_path, adj_model_path, graph_save_path, report_file_path_fix, n_ops = get_inference_result(problem_type, instance_idx, model_type, order, benchmark_instance)
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    print("herkafaen")
    print(inference_assignments.shape)
    print(ops_ma_adj.shape)
    print(target_assignments.shape)
    target_assignments = target_assignments[:, :, :valid_h, :valid_w]
    print(target_assignments.shape)
    print("result")
    # print(inference_assignments)
    #zero_cols = zero_only_columns(inference_assignments)
    #print(f"zeo cols: {zero_cols}")
    print("show")
    print(n_ops)
    print(inference_assignments.shape)



    '''
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis"
    matrix_graph_schedule.show_sched(2, ops_ma_adj, inference_assignments, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, save_path)
    '''

    print("counting infeasibilities")
    inference_assignments_order = code.inference.inferenced_to_schedule.show_order_clear(inference_assignments, n_ops, ops_ma_adj, r_global=False) # tidligere order visning ... DENNE ER KLART BEDRE, far langt mindre feil for mk01
    # inference_assignments_order = code.inference.inferenced_to_schedule.fix_x(inference_assignments, ops_ma_adj) # mk10 order visning
    print('n ops... ')
    print(n_ops)
    report, total_errors, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = code.inference.report_infeasibilities.count_infeasibilities(inference_assignments_order, td["ops_sequence_order"][:valid_w], report_file_path=report_file_path, n_ops=n_ops)
    code.inference.report_infeasibilities.add_elapsed_time_report(elapsed, report_file_path)


    '''
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis"
    matrix_graph_schedule.show_single_sched(inference_assignments_order, 2, td["ops_sequence_order"], ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.8, show_values=True)
    '''
    #save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis"
    #assignments_over_time.append(inference_assignments_order)
    #matrix_graph_schedule.show_sched(2, ops_ma_adj, inference_assignments_order, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, save_path)



    print("2")
    print("fixed infeasibilities")
    inference_assignments_fixed = code.inference.infeasibilities.fix_infeas_mk10(inference_assignments_order, ops_ma_adj, td["ops_sequence_order"], 
                                                                                 valid_w=valid_w, n_ops=n_ops, td=td, analyse_infeas=False) # Bruker denne .. inkluder bare de nedre argumenta hvis skal analysere
    # inference_assignments_fixed = code.inference.infeasibilities.fix_infeasibilities(inference_assignments, ops_ma_adj, td["ops_sequence_order"], inference_assignments_order, n_ops)
    inference_assignments_order = code.inference.inferenced_to_schedule.show_order_clear(inference_assignments_fixed, n_ops, ops_ma_adj)
    
    print("after fixing everything... finally")
    report_fix, total_errors_fix, error_list, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix = code.inference.report_infeasibilities.count_infeasibilities(
    inference_assignments_order,
    td["ops_sequence_order"][:valid_w],
    report_file_path=report_file_path_fix,
    n_ops=n_ops
    )


    print('exiting')
    exit()

    # save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis_fix"
    # matrix_graph_schedule.show_sched(2, ops_ma_adj, inference_assignments, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, save_path)
    '''
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis_fix"
    matrix_graph_schedule.show_single_sched(inference_assignments_fixed, 2, td["ops_sequence_order"], ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.8, show_values=True)
    '''

    """
    index = 0
    for key, value in report.items():
        if report[key]["seq"] != 0:
            break
        index += 1

    print(f"using index {index}")
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/evolution"
    utils.show_single_sched(inference_assignments, index, td["ops_sequence_order"], 
                            ops_ma_adj, valid_h, valid_w, 
                            save_path, n_ops, cell_size=1, show_values=True)

    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/evolution2"
    utils.show_single_sched(inference_assignments_fixed, index, td["ops_sequence_order"], 
                            ops_ma_adj, valid_h, valid_w, 
                            save_path, n_ops, cell_size=1, show_values=True)

   
    print("3")
    """

    # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESUeckT
    # utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order, ops_ma_adj, valid_h, valid_w)

    # SCHEDULING THE INFERENCED SCHEDULE
    # TODO endre hvis kjorer mk10
    # n_machines = 6 # mk01
    n_machines = valid_h
    print(f'n machines... {n_machines}')
    td_scheduled, min_makespan, max_makespan, avg_makespan = schedule.schedule_from_inference(inference_assignments_order, order, env, td.copy(), graph_save_path, n_jobs, n_machines, error_list, ops_sequence_order=td["ops_sequence_order"], report_file_path=report_file_path,
                                                                                              fill_gaps=True)
    
    code.inference.report_infeasibilities.add_makespans_report(min_makespan, max_makespan, avg_makespan, report_file_path)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)


    # TODO beste makespan, avg makespan, verste makespan, avg infeas ... infeas er ogsa knutta til batch storrelse sa tar ogsa total bare sa jeg har det, mengde schedules fiksa av fiksealgo, avg runtime for a produsere schedules
    #min_makespan, avg_makespan, max_makespan, elapsed
    # fur fiksa sched
    #total_error, avg_amt, infeas_ops, avg_amt_infeas_multi, avg_amt_infeas_seq,    avg_infeas_ops, avg_infeas_multi, amt_infeas_sched, percent_infeas_sched
    # etter fiksa sched
    #total_error_fix, avg_amt_fix, infeas_ops_fix, avg_amt_infeas_multi_fix, avg_amt_infeas_seq_fix, avg_infeas_ops_fix, avg_infeas_multi_fix, amt_infeas_sched_fix, percent_infeas_sched_fix

    # min_makespan, avg_makespan, max_makespan, elapsed,
    # report_fix, total_errors_fix, error_list, total_error_fix, avg_amt_infeas_ops_fix, avg_amt_infeas_multi_fix, avg_amt_infeas_seq_fix, infeas_ops_fix, avg_infeas_multi_fix, amt_infeas_sched_fix, percent_infeas_sched_fix
    # report, total_errors, error_list, total_error, avg_amt_infeas_ops, avg_amt_infeas_multi, avg_amt_infeas_seq, infeas_ops, avg_infeas_multi, amt_infeas_sched, percent_infeas_sched

    '''
    benchmark = 'mk02'
    confidence_interval_utils.append_results( 
        min_makespan, avg_makespan, max_makespan, elapsed,
        total_errors, total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,
        total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,
        benchmark
    )
    '''
   



   




# HER
def compare_inference(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    w = 60
    h = 10
    n_jobs = 10

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)

    env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
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


    target_assignments, proc_times, job_ops_adj, ops_ma_adj = dataset_utils.get_dataset_features(td, env, order)

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
    inference_assignments = code.inference.inferenced_to_schedule.round_to_values(inference_assignments, 16, ops_ma_adj)
    comparison_inference_assignments = code.inference.inferenced_to_schedule.round_to_values(comparison_inference_assignments, 16, ops_ma_adj)
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
    td_scheduled = schedule.schedule_from_inference(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)
    td_scheduled_comp = schedule.schedule_from_inference(comparison_inference_assignments, order, env, td.copy(), graph_save_path_comp, n_jobs)

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

# dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
# dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/'
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
instance_i = 500
w = 16
h = 4
n_jobs = 4


benchmark = "mk03"
ins = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{benchmark}.txt'
# ins = None
get_inference_result_cached(model_type, order, instance_idx, w, h, n_jobs, benchmark, ins)
# exit()

