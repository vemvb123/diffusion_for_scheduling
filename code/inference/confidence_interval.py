import csv
import os
import pandas as pd

def append_results(csv_path,
        min_makespan, avg_makespan, max_makespan, elapsed,
        total_errors, total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,
        total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,
        benchmark
    ):
    

    fieldnames = [
        "min_makespan", "avg_makespan", "max_makespan", "elapsed", 

        # before fixing schedule
        "total_errors", "total_error", "total_error_p", "multi_p", "seq_p",
        "infeas_rate", "multi_rate", "seq_rate", "amf_infeas", "amt_infeas_p",

        # after fixing schedule
        "total_errors_fix", "total_error_fix", "total_error_p_fix", "multi_p_fix", "seq_p_fix",
        "infeas_rate_fix", "multi_rate_fix", "seq_rate_fix", "amf_infeas_fix", "amt_infeas_p_fix"
    ]

    # Write header if file doesn't exist
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)

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
            infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix
        ])



def compute_mean():
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'
    csv_bench = 'batch_runs_metrics_mk01_guide_gamma1.csv'
    df = pd.read_csv(f"{folder}/{csv_bench}")
    field = 'total_error'

    mean_value_gamma = df[field].mean()

    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'
    csv_bench = 'batch_runs_metrics_mk01_ddpm.csv'
    df = pd.read_csv(f"{folder}/{csv_bench}")

    mean_value_ddpm = df[field].mean()

    print(mean_value_gamma)
    print(mean_value_ddpm)

# compute_mean()


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

def test_better():
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'

    field = 'total_error'
    field2 = 'min_makespan'

    # Gamma
    gamma_df = pd.read_csv(f"{folder}/batch_runs_metrics_mk01_guide_gamma1.csv")

    # DDPM
    ddpm_df = pd.read_csv(f"{folder}/batch_runs_metrics_mk01_ddpm.csv")

    # Extract columns
    gamma = gamma_df[field]
    ddpm = ddpm_df[field]

    # Means
    mean_gamma = gamma.mean()
    mean_ddpm = ddpm.mean()

    # Min makespan means
    mean_makespan_gamma = gamma_df[field2].mean()
    mean_makespan_ddpm = ddpm_df[field2].mean()

    # Improvement
    absolute_improvement = mean_ddpm - mean_gamma
    percent_improvement = 100 * absolute_improvement / mean_ddpm

    # Welch t-test
    t_stat, p_value = ttest_ind(ddpm, gamma, equal_var=False)

    # Cohen's d
    n1, n2 = len(ddpm), len(gamma)

    var1 = np.var(ddpm, ddof=1)
    var2 = np.var(gamma, ddof=1)

    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )

    cohens_d = (mean_ddpm - mean_gamma) / pooled_std

    # Print
    print("=== total_error ===")
    print(f"Gamma mean: {mean_gamma:.6f}")
    print(f"DDPM mean:  {mean_ddpm:.6f}")

    print(f"\nExpected improvement:")
    print(f"  Absolute: {absolute_improvement:.6f}")
    print(f"  Percent:  {percent_improvement:.2f}%")

    print(f"\nStatistical test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.6e}")

    print(f"\nEffect size:")
    print(f"  Cohen's d:   {cohens_d:.4f}")

    print("\n=== min_makespan ===")
    print(f"Gamma mean min_makespan: {mean_makespan_gamma:.6f}")
    print(f"DDPM mean min_makespan:  {mean_makespan_ddpm:.6f}")


# test_better()


def compute_bounds():
    # Load all stored metrics from CSV
    folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/'
    csv_bench = 'batch_runs_metrics_mk01_guide_gamma1.csv'
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
    bounds_df.to_csv(f"{folder}/metric_bounds_95.csv")

    print("Saved 95% bounds for all metrics to metric_bounds_95.csv")


# compute_bounds()