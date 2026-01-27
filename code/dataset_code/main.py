import code.dataset_code.utils as utils


def main():
    ### Lag dataset


    test_size = 20000
    train_size = 100000
    n = train_size + test_size



    ## Lag datasett for størrelse 444
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444'
    utils.make_dataset(
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

    ## Lag datasett for brandimarte instanse mk01
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
    parameters = utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)

    utils.make_dataset(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
        max_eligable_ma_per_op=parameters['max_machine_options'],
        min_eligable_ma_per_op=parameters['min_machine_options'],
        target_model='/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt',
        order=True,
    )
    print("Done making mk01 dataset")

main()