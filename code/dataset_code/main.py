import code.dataset_code.benchmark_utils
import code.dataset_code.dataset_utils as dataset_utils
import code.scheduling.schedule as schedule
from code.dataset_code.dataset import Dataset_RL4CO

import sys


def main(instance_type = None):
    ### Lag dataset


    train_size = 300000
    test_size = int(train_size * 0.2)
    n = train_size + test_size



    ## Lag datasett for størrelse 444
    if instance_type == "444":

        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444'
        dataset_utils.make_dataset(
            n, dataset_folder,
            n_jobs=4,
            n_ma=4,
            max_op_per_job=4,
            min_op_per_job=4,
            max_proc_time=50,
            min_proc_time=5,
            max_eligable_ma_per_op=4,
            min_eligable_ma_per_op=4,
            target_model='/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001.ckpt',
            order=True,
        )
        print("done making 444")
        exit()

    dataset_folder = None
    filepath_benchmark_instance = None
    target_model = None


    if int(sys.argv[1]) == 1:
        print("making mk15 dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk15_30j_15ma_11op'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk15.txt'
        # TODO har ikke target model fyllt inn her...

    elif int(sys.argv[1]) == 2:
        print("making mk10 dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk10_20j_15ma_14op'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
        target_model = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.001_20j_15ma_14op_mk10.ckpt'

    elif int(sys.argv[1]) == 3:
        print("making 18a dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_18a_20j_10ma_25op'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/dauzere/18a.txt'
        target_model = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.001_20j_10ma_25op_18a.ckpt'

    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_benchmark_instance)
    print(parameters)

    print("stated making dataset")
    print(f"dataset folder: {dataset_folder}")
    print(f"benchmark instance: {filepath_benchmark_instance}")
    print(f"target model: {target_model}")

    dataset_utils.make_dataset(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
        max_eligable_ma_per_op=parameters['max_machine_options'],
        min_eligable_ma_per_op=parameters['min_machine_options'],
        target_model=target_model,
        order=True
    )
    print("Done making mk00 dataset")




def check_benchmark_parameters():
    #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/'
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
    #filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/dauzere/18a.txt'
    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)


def check_dataset():
    instance_idx = 10
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk10_20j_15ma_14op'
    td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)

    print(td.keys())
    print(td['opt_assignment'])
    print(td['opt_actions'])
    print(td['opt_assignment_order'])
    print(td['opt_assignment_order'].shape)
    
    """
    filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk15.txt'
    
    generator_params = {
        'n_jobs':4,
        'n_machines':4,
        'max_op_per_job':4,
        'min_op_per_job':4,
        'max_proc_time':50,
        'min_proc_time':5,
        'max_eligable_ma_per_op':4,
        'min_eligable_ma_per_op':4,
    }


    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444'
    dataset = Dataset_RL4CO(
        folder=dataset_folder,
        generator_params=generator_params,
        order=True,
        w=24,
        h=24
    )
    dataset[0]
    """

#check_benchmark_parameters()

# check_dataset()
#instance_type = "mk01"
main()