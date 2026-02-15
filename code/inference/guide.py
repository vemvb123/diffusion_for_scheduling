import torch

import code.inference.utils as utils

# example ... Error for how much lastlast  Machine is used
# assumes no order given in schedule
def guiding_function(x): #  is batch of instances
    # x has shape [1, 1, 20, 20]
    # extract the 4×16 region
    x_inside = x[:, :, :4, :16]   # shape [1, 1, 4, 16]

    # select the bottom row of that 4×16 → index 3
    bottom_row = x_inside[:, :, 3, :]  # shape [1, 1, 16]

    # compute mean absolute value of bottom row
    error = torch.abs(bottom_row).mean()

    return error



def amt_errors(x, n_ops, ops_ma_adj, ops_seq_order, valid_h, valid_w, max_allowed_total_errors=30):
    # remove padding
    x = x[:, :, :valid_h, :valid_w]
    print("hvata")
    print(x.shape)
    print("---")
    # make in schedule with order
    inference_assignments = utils.show_order_clear(x, n_ops, ops_ma_adj)
    # get amount of errors
    report, total_errors, error_list = utils.assert_sequence_respected(inference_assignments, ops_seq_order)
    # sammenligner med max mengde tillate feil
    error = max_allowed_total_errors - total_errors
    # hvis flere enn max_allowed_total_errors, er error 1
    if error <= total_errors:
        return float(1)
    # normaliserer totale feil mot max_allowed_total_errors
    norm_error = (error) / (max_allowed_total_errors)
    # returnerer error
    return norm_error





# få til clear order, men kan også ta binær, binær er sikkert raskere
# map assignments til proctid
# summer proc tider for hver maskin
# ta loss som forskjell i proc tider: 
# beste tilfelle: (alle summer, som en sum) / mengde maskiner
# faktisk tilfelle: SUM( abs(sumN - beste) )
# normaliser: beste tilfelle, værste tilfelle (beste+beste/2)


def similair_MU(x, h, w, n_ops, ops_ma_adj, proc_times, n_ma):
    """
    Machine-utilization similarity loss.
    Lower is better (0 = perfectly balanced).
    """

    # crop to active region
    x = x[:, :, :h, :w]
    proc_times = proc_times[:, :, :h, :w]
    # binary assignment matrix
    x_used_assignments = utils.round_to_values(x, w, ops_ma_adj)
    # map processing times via assignment
    valid_proc_times = x_used_assignments * proc_times

    # sum processing times per machine
    proc_time_per_machine = valid_proc_times.sum(dim=-1)  # (B, 1, n_ma)
    # total processing time
    total_proc_time = proc_time_per_machine.sum(dim=-1, keepdim=True)  # (B, 1, 1)

    # best possible balanced load
    best_case = total_proc_time / n_ma  # (B, 1, 1)
    # actual deviation from balance
    deviation = torch.abs(proc_time_per_machine - best_case)
    loss_raw = deviation.sum(dim=-1)  # (B, 1)
    # worst reasonable case
    worst_case = best_case * 1.5  # (B, 1, 1)

    # normalized loss
    loss_norm = loss_raw / (worst_case.squeeze(-1) + 1e-8)

    return loss_norm.squeeze(-1)




