from code.scheduling.schedule import make_instance, make_target


def main():

    env, td, generator_params = make_instance(
                    n_jobs=10,
                    n_ma=6,
                    min_proc_time=1,
                    max_proc_time=6,
                    min_op_per_job=5,
                    max_op_per_job=6,
                    min_eligable_ma_per_op=1,
                    max_eligable_ma_per_op=3,
                    batch_size=1
                    )
    # ma .. 6max op, n jobs 10 .... så for hver ma, er alle mulig op der...
    # for en op, hvordan vise at en op ikke kan skeduleres til den ma-en?
    checkpoint_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt"
    td_scheduled, actions = make_target(env, td.copy(), checkpoint_path)

    for key in td.keys():
        print(key)

    print("seq order")
    print(td["ops_sequence_order"])  # gir rekkefølgen av operasjonene
    print(td["ops_sequence_order"].shape)
    print("job ops adj")
    print(td["job_ops_adj"])  # gir om en op tilhører jobben... conditionerer på denne også
    print(td["job_ops_adj"].shape)
    print("ops ma adj")
    print(td["ops_ma_adj"]) # denne gir om ma kan benytte op, kan lett conditione på den.
    print(td["ops_ma_adj"].shape)
    print("proc_times")
    print(td["proc_times"])
    print(f"Actions: {actions}")
    print("scheduled:")
    print(td_scheduled["ma_assignment"])
    print(td_scheduled["ma_assignment"].shape)