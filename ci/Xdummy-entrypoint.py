#!/usr/bin/python3

"""This script starts an X server and sets DISPLAY, then runs wrapped command."""

# Usage: ./Xdummy-entrypoint.py [command]
#
# Adapted from https://github.com/openai/mujoco-py/blob/master/vendor/Xdummy-entrypoint
# Copyright OpenAI; MIT License

import argparse
import os
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extra_args = parser.parse_known_args()

    subprocess.Popen(
        [
            "nohup",
            "Xorg",
            "-noreset",
            "+extension",
            "GLX",
            "+extension",
            "RANDR",
            "+extension",
            "RENDER",
            "-logfile",
            "/tmp/xdummy.log",
            "-config",
            "/etc/dummy_xorg.conf",
            ":0",
        ],
    )
    os.environ["DISPLAY"] = ":0"

    if not extra_args:
        argv = ["/bin/bash"]
    else:
        argv = extra_args

    # Explicitly flush right before the exec since otherwise things might get
    # lost in Python's buffers around stdout/stderr (!).
    sys.stdout.flush()
    sys.stderr.flush()

    os.execvpe(argv[0], argv, os.environ)
