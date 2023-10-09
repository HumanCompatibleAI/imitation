r"""Generate commands to run training scripts with different configs.

This program takes as input a set of config files and outputs a command
for each config file. Each command runs a training script using the
associated config file. By default, each command uses the same random seed
for training. However, if multiple random seeds are specified, the program
generates multiple commands for each config file (one for each random seed).
The commands can be executed by copying and pasting them into a command line
or by piping them to another command line utility. This program is helpful
for running a large number of training scripts with different config files
using multiple random seeds.

For example, we can run:

TUNED_HPS_DIR=../src/imitation/scripts/config/tuned_hps
python commands.py \
    --name=run0 \
    --cfg_pattern=$TUNED_HPS_DIR/*ai*_seals_walker_*.json \
    --output_dir=output

And get the following commands printed out:

python -m imitation.scripts.train_adversarial airl \
    --capture=sys --name=run0 \
    --file_storage=output/sacred/$USER-cmd-run0-airl-0-a3531726 \
    with ../src/imitation/scripts/config/tuned_hps/airl_seals_walker_best_hp_eval.json \
    seed=0 logging.log_root=output

python -m imitation.scripts.train_adversarial gail \
    --capture=sys --name=run0 \
    --file_storage=output/sacred/$USER-cmd-run0-gail-0-a1ec171b \
    with $TUNED_HPS_DIR/gail_seals_walker_best_hp_eval.json \
    seed=0 logging.log_root=output

We can execute commands in parallel by piping them to GNU parallel:

python commands.py ... | parallel -j 8

If the --remote flag is enabled, then the program prints out commands
to run training scripts in containers on the Hofvarpnir cluster.

For example, we can run:

TUNED_HPS_DIR=../src/imitation/scripts/config/tuned_hps
python commands.py \
    --name=run0 \
    --cfg_pattern=$TUNED_HPS_DIR/bc_seals_half_cheetah_best_hp_eval.json \
    --output_dir=/data/output \
    --remote

And get the following command printed out:

ctl job run --name $USER-cmd-run0-bc-0-72cb1df3 \
    --command "python -m imitation.scripts.train_imitation bc \
    --capture=sys --name=run0 \
    --file_storage=/data/output/sacred/$USER-cmd-run0-bc-0-72cb1df3 with \
    /data/imitation/src/imitation/scripts/config/tuned_hps/
    bc_seals_half_cheetah_best_hp_eval.json \
    seed=0 logging.log_root=/data/output" \
    --container hacobe/devbox:imitation \
    --login --force-pull --never-restart --gpu 0 --shared-host-dir-mount /data
"""
import argparse
import glob
import os
import zlib

_ALGO_NAME_TO_SCRIPT_NAME = {
    "bc": "train_imitation",
    "dagger": "train_imitation",
    "airl": "train_adversarial",
    "gail": "train_adversarial",
}

_CMD_ID_TEMPLATE = "$USER-cmd-{name}-{algo_name}-{seed}-{cfg_id}"

_TRAIN_CMD_TEMPLATE = """python -m imitation.scripts.{script_name} \
{algo_name} --capture=sys --name={name} --file_storage={file_storage} \
with {cfg_path} seed={seed} logging.log_root={log_root}"""

_HOFVARPNIR_CLUSTER_CMD_TEMPLATE = """ctl job run \
--name {name} --command "{command}" --container {container} \
--login --force-pull --never-restart --gpu 0 \
--shared-host-dir-mount /data"""


def _get_algo_name(cfg_file: str) -> str:
    """Get the algorithm name from the given config filename."""
    algo_names = set()
    for key in _ALGO_NAME_TO_SCRIPT_NAME:
        if cfg_file.find(key + "_") != -1:
            algo_names.add(key)

    if len(algo_names) == 0:
        raise ValueError("Unable to find algo_name in cfg_file: " + cfg_file)

    if len(algo_names) >= 2:
        raise ValueError("algo_name is ambiguous in cfg_file: " + cfg_file)

    algo_name = algo_names.pop()
    return algo_name


def _get_cfg_id(cfg_path: str) -> str:
    """Get an ID for the config from the given config path."""
    checksum = zlib.adler32(cfg_path.encode())
    checksum_hex = hex(checksum)
    assert checksum_hex.startswith("0x")
    return checksum_hex[2:]


def main(args: argparse.Namespace) -> None:
    """Generate commands to run training scripts with different configs."""
    cfg_relative_paths = glob.glob(args.cfg_pattern)
    local = not args.remote

    for cfg_relative_path in cfg_relative_paths:
        cfg_file = os.path.basename(cfg_relative_path)
        algo_name = _get_algo_name(cfg_file)
        script_name = _ALGO_NAME_TO_SCRIPT_NAME[algo_name]

        if local:
            cfg_path = cfg_relative_path
        else:
            cfg_path = os.path.join(args.remote_cfg_dir, cfg_file)

        cfg_id = _get_cfg_id(cfg_file)

        for seed in args.seeds:
            cmd_id = _CMD_ID_TEMPLATE.format(
                name=args.name,
                algo_name=algo_name,
                seed=seed,
                cfg_id=cfg_id,
            )

            file_storage = os.path.join(args.output_dir, "sacred", cmd_id)

            train_cmd = _TRAIN_CMD_TEMPLATE.format(
                script_name=script_name,
                algo_name=algo_name,
                name=args.name,
                cfg_path=cfg_path,
                seed=seed,
                file_storage=file_storage,
                log_root=args.output_dir,
            )

            if local:
                print(train_cmd)
                continue

            # Escape double quotes.
            command = train_cmd.replace('"', '\\"')

            hofvarpnir_cluster_cmd = _HOFVARPNIR_CLUSTER_CMD_TEMPLATE.format(
                name=cmd_id,
                command=command,
                container=args.container,
            )

            print(hofvarpnir_cluster_cmd)


def parse() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate commands to run training scripts with different configs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="""A name that identifies multiple commands as \
coming from the same 'run'. In particular, this flag is passed to imitation training \
scripts directly in the --name flag and as part of the path in the \
--file_storage flag. If the --remote flag is enabled, this flag is also used in the \
cluster job name.""",
    )
    parser.add_argument(
        "--cfg_pattern",
        type=str,
        default="bc_seals_half_cheetah_best_hp_eval.json",
        help="""Generate a command for every file that matches this glob pattern. \
Each matching file should be a config file that has its algorithm name \
(bc, dagger, airl or gail) bookended by underscores in the filename. \
If the --remote flag is enabled, then generate a command for every file in the \
--remote_cfg_dir directory that has the same filename as a file that matches this \
glob pattern. E.g., suppose the current, local working directory is 'foo' and \
the subdirectory 'foo/bar' contains the config files 'bc_best.json' and \
'dagger_best.json'. If the pattern 'bar/*.json' is supplied, then globbing \
will return ['bar/bc_best.json', 'bar/dagger_best.json']. \
If the --remote flag is enabled, 'bar' will be replaced with `remote_cfg_dir` and \
commands will be created for the following configs: \
[`remote_cfg_dir`/bc_best.json, `remote_cfg_dir`/dagger_best.json] \
Why not just supply the pattern '`remote_cfg_dir`/*.json' directly? \
Because the `remote_cfg_dir` directory may not exist on the local machine.""",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Space-delimited list of random seeds to use for each config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="""Directory in which to store the output training. \
If the --remote flag is enabled, this directory should be accessible \
from each container, e.g., '/data/output/' if /data is the shared directory.""",
    )
    parser.add_argument(
        "--remote",
        default=False,
        action="store_true",
        help="""Generate commands to run training scripts \
in containers on the Hofvarpnir cluster.""",
    )
    # The following flags are only used when the --remote flag is enabled.
    # Otherwise, they are ignored.
    parser.add_argument(
        "--remote_cfg_dir",
        type=str,
        default="/data/imitation/src/imitation/scripts/config/tuned_hps",
        help="""Path to a directory storing config files \
accessible from each container. """,
    )
    parser.add_argument(
        "--container",
        type=str,
        default="hacobe/devbox:imitation",
        help="The image name to use for the containers.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    main(args)
