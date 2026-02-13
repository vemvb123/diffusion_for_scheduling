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







def similair_MU(x, h, w, n_ops, ops_ma_adj, proc_times, n_ma):


    x = x[:, :, :h, :w]
    proc_times = proc_times[:, :, :h, :w]

    x_schedule_order = utils.show_order_clear(x, n_ops, ops_ma_adj, r_global=True)

    sum_over_all = 0
    for i in range(n_ops):
        sum_ma = 0
        first_op_in_ma_proc = 0
        for i in range(n_ma):

            if i == n_ops:
                break

            idx = torch.nonzero(x_schedule_order == i, as_tuple=False)
            row, col = idx[0]
            proc_time = proc_times[row, col]
            sum_ma += proc_time

            if first_op_in_ma_proc == 0:
                first_op_in_ma_proc = proc_time

        sum_ma / n_ma
        difference = abs(sum_ma - first_op_in_ma_proc)
        sum_over_all += difference