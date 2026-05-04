

# benchmark_instance = 'mk01' for example
def get_benchmark_values(benchmark_instance: str):
    if benchmark_instance == "mk02":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.001_10j_6ma_6op_mk02.ckpt'
        model_path = None
        dataset_train_path = None 
        dataset_test_path = None
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk02.txt'
        valid_h = None
        valid_w = None
        w = None
        h = None 
    if benchmark_instance == "mk05":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_15j_4ma_9op_mk05.ckpt'
        model_path = None
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_4ma_9op_mk05'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk05.txt'
        valid_h = None
        valid_w = None
        w = None
        h = None 
    if benchmark_instance == "mk06":

        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_10ma_15op_mk06.ckpt'
        model_path = None
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_10j_10ma_15op_mk06'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk06.txt'
        valid_h = None
        valid_w = None
        w = None
        h = None 
        name = "10j_10ma_15op_mk06"

    if benchmark_instance == "mk10":
        checkpoint_path = None
        model_path = None
        dataset_train_path = None
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
        valid_h = None
        valid_w = None
        w = None
        h = None 
        name = "20j_15ma_14op_mk10"





    else:
        raise ValueError(f"{benchmark_instance} not supported")




    return checkpoint_path, model_path, dataset_train_path, dataset_test_path, filepath_benchmark_instance, valid_h, valid_w, w, h, name

