from __future__ import annotations

import pickle
from typing import List

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np


def matrix_is_symmetric(a: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Returns True if the provided matrix is symmetric and False otherwise.

    https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy

    Args:
        a: The array to check.
        rtol: The relative tolerance.
        atol: The absolute tolerance.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol, equal_nan=True)


class DistanceMatrix:
    """Represents a matrix of distance values.

    This class mostly exists for visualization purposes.

    Args:
        labels: The ordered labels associated with the cols / rows of the distances.
        distances: The matrix of distance values.
    """
    def __init__(self, labels: List[str], distances: np.ndarray):
        self.labels = labels
        assert distances.ndim == 2
        assert matrix_is_symmetric(distances), f"Expected symmetric matrix, but got {distances}"
        self.distances = distances

    def distance_between(self, label_1: str, label_2: str) -> float:
        """Returns the distance between two rewards.

        Args:
            label_1: The first reward label.
            label_2: The second reward label.

        Returns:
            The distance between the requested reward labels. Asserts they exist.
        """
        for label in [label_1, label_2]:
            assert label in self.labels, f"Missing label: {label}, available labels: {self.labels}"
        index_1 = self.labels.index(label_1)
        index_2 = self.labels.index(label_2)
        return self.distances[index_1, index_2]

    def visualize(
            self,
            filepath: str,
            title: str,
            vmin: float = 0,
            vmax: float = 1,
            cmap_str: str = "Reds",
            fontsize: int = 10,
            height_per_cell: float = 1,
            width_per_cell: float = 4,
    ) -> None:
        """Visualize this distance matrix as a table of distances.
••••••••
        Args:
            filepath: Where to save the figure.
            title: The title to use for the figure.
        """
        vmax = max(vmax, self.distances.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.ScalarMappable(cmap=cmap_str, norm=norm)

        num_labels = len(self.labels)

        table_text = []
        table_colors = []
        for i in range(num_labels):
            text_row = []
            color_row = []
            for j in range(num_labels):
                distance = self.distances[i, j]
                formatted_distance = f"{distance:0.4f}"
                text_row.append(formatted_distance)
                color_row.append(cmap.to_rgba(distance))
            table_text.append(text_row)
            table_colors.append(color_row)

        fig_width = max(4, width_per_cell * num_labels)
        fig_height = max(4, height_per_cell * num_labels)
        plt.figure(figsize=(fig_width, fig_height))
        col_widths = [.2] * num_labels
        table = plt.table(
            cellText=table_text,
            cellColours=table_colors,
            rowLabels=self.labels,
            colLabels=self.labels,
            colWidths=col_widths,
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 2)

        ax = plt.gca()
        ax.axis("off")
        plt.title(title, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def save(self, filepath: str) -> None:
        """Saves the contents of this class to file."""
        with open(filepath, "wb") as outfile:
            pickle.dump(self.__dict__, outfile)

    @classmethod
    def load(cls: type, filepath: str) -> DistanceMatrix:
        """Loads this class from file."""
        with open(filepath, "rb") as infile:
            data = pickle.load(infile)
            return cls(data["labels"], data["distances"])