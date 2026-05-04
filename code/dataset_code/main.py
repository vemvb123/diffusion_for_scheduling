import code.dataset_code.benchmark_utils
import code.dataset_code.dataset_utils as dataset_utils
import code.scheduling.schedule as schedule
from code.dataset_code.dataset import Dataset_RL4CO
from code.dataset_code import benchmark_utils
import sys
from code.dataset_code.benchmark_values import get_benchmark_values

def main(instance_type = None):
    ### Lag dataset
    print("Beginning dataset creation...")

    train_size = 100000
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

    elif int(sys.argv[1]) == 4:
        print("making mk02 dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_10j_6ma_6op_mk02'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk02.txt'
        target_model = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.001_10j_6ma_6op_mk02.ckpt'

    elif int(sys.argv[1]) == 5:
        print("making mk03 dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_10op_mk03'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk03.txt'
        target_model = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_15j_8ma_10op_mk03.ckpt'

    elif int(sys.argv[1]) == 6:
        print("making mk04 dataset")
        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_9op_mk04'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk04.txt'
        target_model = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_15j_8ma_9op_mk04.ckpt'

    elif int(sys.argv[1]) == 7:
        target_model, _, dataset_folder, _, filepath_benchmark_instance, _, _, _, _ = get_benchmark_values('mk05')

    elif int(sys.argv[1]) == 8:
        target_model, _, dataset_folder, _, filepath_benchmark_instance, _, _, _, _ = get_benchmark_values('mk06')



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
    print("checking benchmark parameters...")
    #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/'
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk06.txt'
    #filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/dauzere/18a.txt'
    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)


def check_dataset():
    # --- process test image
    instance_idx = 10
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_10op_mk03'
    td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)


    print(td['job_ops_adj'])
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
    ):w
    dataset[0]
    """

# mk04 er: 
# w: 8, h: 135 -> 24, 136
# {'n_jobs': 15, 'n_machines': 8, 'min_processing_time': 1, 'max_processing_time': 9, 'fewest_operations': 3, 'most_operations': 9, 'min_machine_options': 1, 'max_machine_options': 3}


# mk05 er: 
# check_benchmark_parameters()

# check_dataset()
#instance_type = "mk01"
main()

# ins = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
# td = benchmark_utils.make_the_stuff(ins)


"""
print(td['proc_times'])
print(td['ops_ma_adj'])
print(td['job_ops_adj'])
print(td['ops_sequence_order'])
#print(td.keys())

print('------------------')

"""
"""
instance_idx = 10
dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)

print(td['proc_times'])
print(td['ops_ma_adj'])
print(td['job_ops_adj'])
print(td['ops_sequence_order'])
print(td.keys())


"""