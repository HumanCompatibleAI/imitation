"""Configuration settings for analyze, inspecting results from completed experiments."""

import os.path as osp

import sacred

analysis_ex = sacred.Experiment("analyze")


@analysis_ex.config
def config():
    # Recursively search in this directory for sacred logs
    source_dir_str = "output/sacred/train_adversarial"
    skip_failed_runs = True  # Skip analysis for logs that have FAILED status
    run_name = None  # Restrict analysis to sacred logs with a certain run name
    env_name = None  # Restrict analysis to sacred logs with a certain env name
    csv_output_path = None  # Write CSV output to this path
    tex_output_path = None  # Write LaTex output to this path
    print_table = True  # Set to True to print analysis to stdout
    split_str = ","  # str used to split source_dir_str into multiple source dirs
    table_verbosity = 1  # Choose from 0, 1, 2 or 3
    source_dirs = None


@analysis_ex.config
def convert_source_dirs(source_dir_str, split_str, source_dirs):
    if source_dirs is None:
        source_dirs = source_dir_str.split(split_str)

    source_dirs = [osp.expanduser(p) for p in source_dirs]
