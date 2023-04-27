#!/usr/bin/env python
"""Convert trajectories from `imitation` format to openai/baselines GAIL format."""

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from imitation.data import rollout, serialize, types


def convert_trajs_to_sb(trajs: Sequence[types.TrajectoryWithRew]) -> dict:
    """Converts Trajectories into the dict format used by Stable Baselines GAIL."""
    trans = rollout.flatten_trajectories_with_rew(trajs)
    return dict(
        acs=trans.acts,
        rews=trans.rews,
        obs=trans.obs,
        ep_rets=np.array([np.sum(t.rews) for t in trajs]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=str)
    parser.add_argument("dst_path", type=str)
    args = parser.parse_args()

    src_path = Path(args.src_path)
    dst_path = Path(args.dst_path)

    assert src_path.is_file()
    src_trajs = serialize.load_with_rewards(src_path)
    dst_trajs = convert_trajs_to_sb(src_trajs)
    os.makedirs(dst_path.parent, exist_ok=True)
    with open(dst_path, "wb") as f:
        np.savez_compressed(f, **dst_trajs)

    print(f"Dumped rollouts to {dst_path}")


if __name__ == "__main__":
    main()
