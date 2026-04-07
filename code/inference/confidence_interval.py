import csv
import os
import pandas as pd


def append_results(
        min_makespan, avg_makespan, max_makespan, elapsed,
        total_errors, total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,
        total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,
        benchmark
    ):
    
    import os, csv

    path_before = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals'
    csv_path = f"{path_before}/batch_runs_metrics_{benchmark}.csv"

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


def compute_bounds():
    # Load all stored metrics from CSV
    df = pd.read_csv("batch_runs_metrics.csv")

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
    bounds_df.to_csv("metric_bounds_95.csv")

    print("Saved 95% bounds for all metrics to metric_bounds_95.csv")


