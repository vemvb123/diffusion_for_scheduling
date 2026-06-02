import csv
import os
import pandas as pd

def append_ma_usage_result(csv_path, ma_usage_count_per_machine):

    columns = [f"ma{i+1}" for i in range(len(ma_usage_count_per_machine))]
    df = pd.DataFrame([ma_usage_count_per_machine.tolist()], columns=columns)

    df.to_csv(
        csv_path,
        mode='a',
        header=not os.path.exists(csv_path),
        index=False
    )


def append_results(csv_path,
        min_makespan, avg_makespan, max_makespan, elapsed,
        total_errors, total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,
        total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,
        benchmark, busy_count
    ):
    

    fieldnames = [
        "min_makespan", "avg_makespan", "max_makespan", "elapsed", 

        # before fixing schedule
        "total_errors", "total_error", "total_error_p", "multi_p", "seq_p",
        "infeas_rate", "multi_rate", "seq_rate", "amf_infeas", "amt_infeas_p",

        # after fixing schedule
        "total_errors_fix", "total_error_fix", "total_error_p_fix", "multi_p_fix", "seq_p_fix",
        "infeas_rate_fix", "multi_rate_fix", "seq_rate_fix", "amf_infeas_fix", "amt_infeas_p_fix",

        "busy_count"
    ]

    # Write header if file doesn't exist
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)

    line_num = None
    with open(csv_path, "r", newline="") as f:
        line_num = sum(1 for _ in f) + 1   # line where new row will be written

    # Append this batch’s results
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            min_makespan, avg_makespan, max_makespan, elapsed, 

            # before fixing schedule
            total_errors, total_error, total_error_p, multi_p, seq_p,
            infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,

            # after fixing schedule
            total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix,
            infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,

            busy_count
        ])

    print(f'appended results to {csv_path}')
    print(f'appended results to line {line_num}')
    print(' ')

'''
def compute_mean():
    benches = ['mk01', 'mk10']
    timesteps = [400, 500, 600, 700, 800, 900, 1000]
    methods = ['ddpm', 'batchrep']
    for b in benches:
        print(b)
        for t in timesteps:
            for m in methods:
                folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/runs_batchrep'
                csv_bench = f'batch_runs_metrics_{b}_{m}_timesteps_{t}.csv'
                df = pd.read_csv(f"{folder}/{csv_bench}")
                fields = ['total_errors']
                for field in fields:
                    mean = df[field].mean()
                    print(f'{m} {t}: {field}: {mean}')
            print()
        print()
            
'''



def compute_mean():
    benches = ['mk01', 'mk02', 'mk03', 'mk04', 'mk04', 'mk05', 'mk06', 'mk07', 'mk08', 'mk09', 'mk10']
    for b in benches:
        print(b)
        folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'
        csv_bench = f'batch_runs_metrics_{b}_ddpm.csv'
        if b == 'mk01' or b == 'mk10':
            csv_bench = f'batch_runs_metrics_{b}_ddpm_rerun.csv'
        df = pd.read_csv(f"{folder}/{csv_bench}")
        fields = ['min_makespan']
        for field in fields:
            mean = df[field].mean()
            print(f'{b}: {field}: {mean}')


def percentage_improvment():
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'
    field = 'total_error'

    # Gamma method
    csv_gamma = 'batch_runs_metrics_mk01_guide_gamma1.csv'
    df_gamma = pd.read_csv(f"{folder}/{csv_gamma}")

    mean_gamma = df_gamma[field].mean()

    # DDPM method
    csv_ddpm = 'batch_runs_metrics_mk01_ddpm.csv'
    df_ddpm = pd.read_csv(f"{folder}/{csv_ddpm}")

    mean_ddpm = df_ddpm[field].mean()

    # Absolute difference
    difference = mean_ddpm - mean_gamma

    # Relative improvement (%)
    percent_improvement = 100 * difference / mean_ddpm

    print(f"Gamma mean: {mean_gamma}")
    print(f"DDPM mean:  {mean_ddpm}")
    print(f"Absolute improvement: {difference}")
    print(f"Percent improvement: {percent_improvement:.2f}%")

# percentage_improvment()


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def test_better(file_original, file_improv, output_file, field1_mean, field2_test):
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals'

    field = field1_mean
    field2 = field2_test

    # Improved model
    file_improv_df = pd.read_csv(f"{folder}/{file_improv}")

    # Original model
    file_original_df = pd.read_csv(f"{folder}/{file_original}")

    # Extract columns
    file_improv_cols = file_improv_df[field]
    file_original_cols = file_original_df[field]

    # Means
    mean_file_improv = file_improv_cols.mean()
    mean_file_original = file_original_cols.mean()

    # Secondary metric means
    mean_makespan_file_improv = file_improv_df[field2].mean()
    mean_makespan_file_original = file_original_df[field2].mean()

    # Improvement
    absolute_improvement = mean_file_original - mean_file_improv
    percent_improvement = 100 * absolute_improvement / mean_file_original

    # Welch t-test
    t_stat, p_value = ttest_ind(
        file_original_cols,
        file_improv_cols,
        equal_var=False
    )

    # Cohen's d
    n1, n2 = len(file_original_cols), len(file_improv_cols)

    var1 = np.var(file_original_cols, ddof=1)
    var2 = np.var(file_improv_cols, ddof=1)

    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )

    cohens_d = (
        mean_file_original - mean_file_improv
    ) / pooled_std

    # Build output string
    output = f"""
        === {field} ===
        Improved mean: {mean_file_improv:.6f}
        Original mean: {mean_file_original:.6f}

        Expected improvement:
        Absolute: {absolute_improvement:.6f}
        Percent:  {percent_improvement:.2f}%

        Statistical test:
        t-statistic: {t_stat:.4f}
        p-value:     {p_value:.6e}

        Effect size:
        Cohen's d:   {cohens_d:.4f}

        === {field2} ===
        Improved mean {field2}: {mean_makespan_file_improv:.6f}
        Original mean {field2}: {mean_makespan_file_original:.6f}
    """

    # Write to file
    output_path = f"{folder}/{output_file}"

    with open(output_path, "w") as f:
        f.write(output)

    print(f"Results written to: {output_path}")


def compute_bounds(csv_bench, output_name):
    # Load all stored metrics from CSV
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals'
    df = pd.read_csv(f"{folder}/{csv_bench}")

    # We want 2.5th and 97.5th percentiles to get a 95% interval
    lower_pct = 0.025
    upper_pct = 0.975

    bounds = {}

    for col in df.columns:
        # Calculate these percentiles using pandas
        lower_val = df[col].quantile(lower_pct)
        upper_val = df[col].quantile(upper_pct)
        bounds[col] = {"lower_95": lower_val, "upper_95": upper_val}

    # Save metric bounds to separate file
    bounds_df = pd.DataFrame(bounds).T
    bounds_df.to_csv(f"{folder}/{output_name}")

    print(f"Saved 95% bounds for all metrics to {output_name}")


'''
types = ['ddpm']
# benches = ['mk02', 'mk03', 'mk05', 'mk06', 'mk07']
benches = ['mk08', 'mk09']

for t in types:
    for b in benches:
        output_name = f'intervall/metric_bounds_95_{b}_{t}.csv'
        csv_bench = f'batch_runs_metrics_{b}_{t}.csv'
        compute_bounds(csv_bench, output_name)
'''
'''
benches = ['mk01', 'mk10']
gamma_setting = ['0.5', '1.0']
for i in range(len(benches)):
    b = benches[i]
    g = gamma_setting[i]
    t_improv = 'guide'
    t_original = 'ddpm'
    file_improv = f'batch_runs_metrics_{b}_{t_improv}_rerun_{t_improv}_{g}.csv'
    file_original = f'batch_runs_metrics_{b}_{t_original}_rerun.csv'
    output_file = f'statistic/statistic_results_{b}_{t_improv}_{t_original}_rerun.csv'
    field1_mean = 'total_errors'
    field2_test = 'min_makespan'

    test_better(file_original, file_improv, output_file, field1_mean, field2_test)
'''


compute_mean()
