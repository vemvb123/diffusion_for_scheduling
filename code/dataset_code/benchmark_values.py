

# benchmark_instance = 'mk01' for example
def get_benchmark_values(benchmark_instance: str):

    if benchmark_instance == "444":
        # Unsure if this is the correct RL model
        checkpoint_path = 'rl4co_model_0.0001.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/444/adj_type_adj_order_True.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = None # No benchmark for this problem size
        w = 24
        h = 24
        valid_h = 4
        valid_w = 16


    elif benchmark_instance == "mk01":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/cos_beta/timestep_1000_cos.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
        w = 64 # unsure if should be actually 60
        h = 24
        valid_w = 55
        valid_h = 6
        name = "10j_6ma_6op_mk01"

    elif benchmark_instance == "mk01_mautil":
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/mk01_mautil.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk01_10j_6ma_6op_mk01_mautil'

        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt'
        w = 64 # unsure if should be actually 60
        h = 24
        valid_w = 55
        valid_h = 6
        name = "10j_6ma_6op_mk01"


    elif benchmark_instance == "mk02":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.001_10j_6ma_6op_mk02.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk02/mk02.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_10j_6ma_6op_mk02'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk02.txt'
        h = 24
        w = 64
        valid_h = 6
        valid_w = 58
        name = "10j_6ma_6op_mk02"

    elif benchmark_instance == "mk03":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_15j_8ma_10op_mk03.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk03/mk03_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_10op_mk03'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk03.txt'
        h = 24
        w = 152
        valid_h = 8
        valid_w = 150
        name = "15j_8ma_10op_mk03"

    elif benchmark_instance == "mk04":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_15j_8ma_9op_mk04.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk04/mk04.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_8ma_9op_mk04'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk04.txt'
        h = 24
        w = 136
        valid_h = 8
        valid_w = 135
        name = "15j_8ma_9op_mk04"


    elif benchmark_instance == "mk05":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_15j_4ma_9op_mk05.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk05/mk05_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_15j_4ma_9op_mk05'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk05.txt'
        valid_h = 4
        valid_w = 135
        w = 136
        h = 24
        name = "15j_4ma_9op_mk05"

    elif benchmark_instance == "mk06":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_10j_10ma_15op_mk06.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk06/mk06_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_10j_10ma_15op_mk06'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk06.txt'
        valid_h = 10
        valid_w = 150
        w = 152
        h = 24
        name = "10j_10ma_15op_mk06"


    elif benchmark_instance == "mk07":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_20j_5ma_5op_mk07.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk07/mk07_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_20j_5ma_5op_mk07'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk07.txt'
        valid_h = 5
        valid_w = 100
        w = 103
        h = 24
        name = "20j_5ma_5op_mk07"


    elif benchmark_instance == "mk08":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_20j_10ma_14op_mk08.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk08/mk08_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_20j_10ma_14op_mk08'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk08.txt'
        valid_h = 10
        valid_w = 280
        w = 280
        h = 24
        name = "20j_10ma_14op_mk08"

    elif benchmark_instance == "mk09":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.0001_20j_10ma_14op_mk09.ckpt'
        model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk09/mk09_0.0001.pth'
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_20j_10ma_14op_mk09'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk09.txt'
        valid_h = 10
        valid_w = 280
        w = 280
        h = 24
        name = "20j_10ma_14op_mk09"






    elif benchmark_instance == "mk10":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/RL/rl4co_model_0.001_20j_15ma_14op_mk10.ckpt'
        model_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk10/mk10.pth"
        dataset_train_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_mk10_20j_15ma_14op'
        dataset_test_path = dataset_train_path + '_TEST'
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
        h = 24
        w = 280
        valid_h = 15
        valid_w = 280
        name = "20j_15ma_14op_mk10"
    else:
        raise ValueError(f"{benchmark_instance} not supported")




    return checkpoint_path, model_path, dataset_train_path, dataset_test_path, filepath_benchmark_instance, valid_h, valid_w, w, h, name

