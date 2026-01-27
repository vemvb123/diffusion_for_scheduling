from code.dataset_code.utils import get_rl4co_parameters_from_brandimarte_instance, make_dataset


def main():
    ### Lag dataset
    """
    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'

    test_size = 20000
    train_size = 100000
    n = test_size

    batch_size = 184

    make_dataset(20000)
    """


    ### Dekod parameterverdier for Brandimarte instanse
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
    parameters = get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)


    test_size = 20000
    train_size = 100000
    n = train_size + test_size

    dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/batched_mk01_10j_6ma_6op_mk01'
    make_dataset(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
        target_model='/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt',
        order=True,
    )


main()