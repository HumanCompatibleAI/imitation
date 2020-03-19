#!/usr/bin/env python
"""
Convert trajectories from imitation List[Trajectory] format to transitions.

Then save as npz in the correct format.
"""

from typing import List

import numpy as np

from imitation.util import rollout as ro_util


def convert_trajs_to_sb(trajs: List[ro_util.Trajectory]) -> dict:
  """Converts Trajectories into the dict format used by Stable Baselines GAIL.
  """
  trans = ro_util.flatten_trajectories(trajs)
  return dict(
    acs=trans.acts,
    rews=trans.rews,
    obs=trans.obs,
    ep_rets=np.array([np.sum(t.rews) for t in trajs]),
  )


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=str)
    parser.add_argument("dst_path", type=str)
    args = parser.parse_args()

    src_path = Path(args.src_path)
    dst_path = Path(args.dst_path)

    assert src_path.is_file()
    with open(src_path, "rb") as f:
      src_trajs = pickle.load(f)

    dst_trajs = convert_trajs_to_sb(src_trajs)  # type: ro_util.Trajectory
    os.makedirs(dst_path.parent, exist_ok=True)
    with open(dst_path, "wb") as f:
      np.savez_compressed(f, **dst_trajs)

    print(f"Dumped files to {dst_path}")
