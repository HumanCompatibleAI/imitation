from typing import NamedTuple, Tuple

import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


class MapConfig(NamedTuple):
    xs: Tuple[float, float] = (-0.3, 0.3)
    ys: Tuple[float, float] = (-0.3, 0.3)
    xres: int = 50
    yres: int = 50


def make_heat_map(eval_func, map_config):
    gps = get_dense_gridpoints(map_config)
    vals = np.zeros(map_config.xres * map_config.yres)
    for i, pnt in enumerate(gps):
        vals[i] = eval_func(pnt)
    return predictions_to_heatmap(vals, map_config)


def get_dense_gridpoints(map_config):
    xl = np.linspace(map_config.xs[0], map_config.xs[1], num=map_config.xres)
    yl = np.linspace(map_config.ys[0], map_config.ys[1], num=map_config.yres)
    gridpoints = np.zeros((map_config.xres * map_config.yres, 2))
    for i in range(map_config.xres):
        for j in range(map_config.yres):
            gridpoints[i + map_config.xres * j] = np.array((xl[i], yl[j]))
    return gridpoints


def predictions_to_heatmap(predictions, map_config):
    res = np.zeros((map_config.xres, map_config.yres))
    for i in range(map_config.xres):
        for j in range(map_config.yres):
            res[i, j] = predictions[i + map_config.xres * j]
    res = res / np.max(res)
    return res.T


def make_density_map(paths, map_config):
    xs = np.linspace(map_config.xs[0], map_config.xs[1], num=map_config.xres + 1)
    ys = np.linspace(map_config.ys[0], map_config.ys[1], num=map_config.yres + 1)
    y = paths[:, 0]
    x = paths[:, 1]
    H, xedges, yedges = np.histogram2d(y, x, bins=(xs, ys))
    H = H.astype(np.float)
    H = H / np.max(H)
    return H.T


def plot_maps(combined_list=None, *heatmaps):
    import matplotlib.pyplot as plt

    combined = np.c_[heatmaps]
    if combined_list is not None:
        combined_list.append(combined)
        combined = np.concatenate(combined_list)
    else:
        combined_list = []
    plt.figure()
    plt.imshow(combined, cmap="afmhot", interpolation="none")
    plt.show()
    return combined_list


class TwoDEnv(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip, xbounds, ybounds):
        super(TwoDEnv, self).__init__(model_path=model_path, frame_skip=frame_skip)
        assert isinstance(self.observation_space, Box)
        assert self.observation_space.shape == (2,)
        self.__map_config = MapConfig(
            xs=(xbounds[0], xbounds[1]), ys=(ybounds[0], ybounds[1])
        )

    @property
    def map_config(self):
        return self.__map_config

    @property
    def grid_flat_dim(self):
        return self.map_config.xres * self.map_config.yres

    def make_density_map(self, paths):
        return make_density_map(paths, self.map_config)

    def make_heatmap(self, eval_func):
        return make_heat_map(eval_func, self.map_config)

    def get_dense_gridpoints(self):
        return get_dense_gridpoints(self.map_config)

    def predictions_to_heatmap(self, predictions):
        return predictions_to_heatmap(predictions, self.map_config)
