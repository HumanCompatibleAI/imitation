#!/usr/bin/python3

# Adapted from https://github.com/openai/mujoco-py/blob/master/vendor/Xdummy-entrypoint
# Copyright OpenAI; MIT License

import argparse
import os
import sys
import subprocess

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
        ]
    )
    subprocess.Popen(
        ["nohup", "Xdummy"],
        stdout=open("/dev/null", "w"),
        stderr=open("/dev/null", "w"),
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
