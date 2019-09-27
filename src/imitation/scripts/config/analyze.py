import sacred

analysis_ex = sacred.Experiment("analyze")


@analysis_ex.config
def config():
  source_dir = None  # Recursively search in this directory to find Sacred dirs
  skip_failed_runs = True  # Skips analysis for logs that have FAILED status
  run_name = None  # Restricts analysis to sacred logs with a certain run name
  csv_output_path = None  # Write output CSV to this path
  verbose = True  # Set to True to print analysis to stdout
