import sacred

analysis_ex = sacred.Experiment("analyze")


@analysis_ex.config
def config():
    source_dir = "output/sacred/train"  # Recursively search in this directory
    skip_failed_runs = True  # Skip analysis for logs that have FAILED status
    run_name = None  # Restrict analysis to sacred logs with a certain run name
    env_name = None  # Restrict analysis to sacred logs with a certain env name
    csv_output_path = None  # Write output CSV to this path
    verbose = True  # Set to True to print analysis to stdout
