

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


    return checkpoint_path, model_path, dataset_train_path, dataset_test_path, filepath_benchmark_instance, valid_h, valid_w, w, h

