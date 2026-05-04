def write_report(report, total_errors, error_list, print_report=False, save_as_file=None, n_ops=None, only_results=False):

    amt_feas = 0
    multi = 0
    seq = 0
    sum_multi_rate = 0.0
    sum_seq_rate = 0.0
    for key, value in report.items():

        if value["total"] == 0:
            amt_feas += 1
            case_str = f"{key} .. {value} .. NO ERRORS"
            # if print_report: print(case_str)

        else:
            seq_rate = value["seq"] / n_ops if n_ops is not None else value["seq"]
            multi_rate = value["multi"] / n_ops if n_ops is not None else value["multi"]
            sum_multi_rate += multi_rate
            sum_seq_rate += seq_rate
            case_str = f"{key} .. {value} .. multi rate: {multi_rate:.2f}, seq rate: {seq_rate:.2f}"
            if value["multi"] > 0:
                multi += value["multi"]
            if value["seq"] > 0:
                seq += value["seq"]

            if print_report: print(case_str)

    total_errors_str = f"Total errors: {total_errors} - scheduled multiple times: {multi}, break predecessor constraint: {seq}"
    avg_errors = f"Avg errors per instance: {total_errors / len(report):.2f}, scheduled multiple times: {multi / len(report):.2f}, break predecessor constraint: {seq / len(report):.2f}"


    avg_multi_rate = sum_multi_rate / len(report)
    avg_seq_rate = sum_seq_rate / len(report)
    avg_error_rates = f"Avg error rates: {avg_multi_rate + avg_seq_rate} - avg rate scheduled multi: {avg_multi_rate}, avg rate break predecessor constraint: {avg_seq_rate}"

    total_feas_str = f"Total feasible schedules: {amt_feas} / {len(report)}: {amt_feas / len(report) * 100:.2f}%"

    if only_results:
        return avg_multi_rate + avg_seq_rate

    if print_report:
        print(total_errors_str)
        print(avg_errors)
        print(avg_error_rates)
        print(total_feas_str)

    # write report to file
    if save_as_file is not None:
        with open(save_as_file, "w") as f:
            for key, value in report.items():
                if value["total"] == 0:
                    f.write(f"{key} .. {value} .. NO ERRORS\n")
                else:
                    f.write(f"{key} .. {value}\n")
            f.write(total_errors_str + "\n")
            f.write(avg_errors + "\n")
            f.write(avg_error_rates+ "\n")
            f.write(total_feas_str + "\n")
    # total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p
    return (
        total_errors, 
        total_errors / len(report), 
        multi / len(report), 
        seq / len(report),       
        avg_multi_rate + avg_seq_rate, 
        avg_multi_rate, 
        avg_seq_rate, 
        amt_feas, 
        amt_feas / len(report) * 100
    )

def add_makespans_report(min_makespan, max_makespan, avg_makespan, report_path):
    with open(report_path, "a") as file:
        file.write(f"makespans - avg {avg_makespan}, min {min_makespan}, max {max_makespan}")


def add_elapsed_time_report(elapsed, report_path):
    with open(report_path, "a") as file:
        file.write(f"Seconds to run inference: {elapsed} s")


def count_infeasibilities(ma_seq_matrix, ops_sequence_order, do_print=True, valid_h=None, valid_w=None, report_file_path=None, n_ops=None, only_results=False):
    if do_print != None:
        ma_seq_matrix = ma_seq_matrix[:, :, :valid_h, :valid_w]


    """
    Batch-wise validation of operation sequences.

    Returns:
        report: dict
            {run
                batch_idx: {
                    "zero": int,   # zero-only columns
                    "multi": int,  # multiple non-zeros in a column
                    "dup": int,    # duplicate values in a block
                    "seq": int,    # non-strictly-increasing sequences
                }
            }
    """
    total_feasible_schedules = 0

    # ---- normalize shape ----
    if ma_seq_matrix.dim() == 4:
        # (B, 1, M, C) -> (B, M, C)
        ma_seq_matrix = ma_seq_matrix.squeeze(1)

    B, n_machines, n_columns = ma_seq_matrix.shape
    error_list = [0] * B

    if do_print:
        pass
        # print(f"B IS: {B}")

    ops = ops_sequence_order.tolist()
    n = len(ops)
    assert n == n_columns, "ops_sequence_order must match number of columns"

    # ---- build blocks (once) ----
    blocks = []
    start = 0
    for i in range(1, n):
        if ops[i] < ops[i - 1]:
            blocks.append((start, i))
            start = i
    blocks.append((start, n))

    blocks_to_check = blocks[:-1]  # ignore last block

    # ---- report ----
    report = {}

    # ---- iterate over batch ----
    total_errors = 0
    for b in range(B):
        ma = ma_seq_matrix[b]

        errors = {
            #"zero": 0,
            "multi": 0,
            #"dup": 0,
            "seq": 0,

            "total": 0
        }

        # ---- per-section processing ----
        for section_idx, (start, end) in enumerate(blocks_to_check):
            block_values = []

            for col in range(start, end):
                col_vals = ma[:, col]
                nonzeros = col_vals[col_vals > 0]

                if nonzeros.numel() == 0:
                    #errors["zero"] += 1
                    #errors["total"] += 1
                    #error_list[b] += 1
                    #total_errors += 1
                    block_values.append(None)

                elif nonzeros.numel() > 1:
                    errors["multi"] += 1
                    errors["total"] += 1
                    error_list[b] += 1
                    total_errors += 1
                    block_values.append("MULTI")

                else:
                    block_values.append(int(nonzeros.item()))

            if do_print:
                printable = [v if isinstance(v, int) else str(v) for v in block_values]
                # print(f"B {b} Section {section_idx}: {printable}")

            # ---- section-level checks ----
            clean_vals = [v for v in block_values if isinstance(v, int)]
            """
            if len(clean_vals) != len(set(clean_vals)):
                errors["dup"] += 1
                errors["total"] += 1
                error_list[b] += n
                total_errors += 1
            """
            for i in range(1, len(clean_vals)):
                if clean_vals[i] <= clean_vals[i - 1]:
                    errors["seq"] += 1
                    errors["total"] += 1
                    error_list[b] += 1
                    total_errors += 1
                    break
        if do_print:
            pass
            # print(" ")

        # ---- store only failing batches ----
        report[b] = errors

    if only_results:
        print('use only res')
        total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = write_report(report, total_errors, error_list, print_report=do_print, save_as_file=report_file_path, n_ops=n_ops)
        return report, total_errors, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p
    total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = write_report(report, total_errors, error_list, print_report=do_print, save_as_file=report_file_path, n_ops=n_ops)
    return report, total_errors, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p



