import abc
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Dataset(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def sample(self: "Dataset[T]", n_samples: int) -> T:
        """Return a batch of data.

        Args:
            n_samples: A positive integer indicating the number of samples to return.
        Raises:
            ValueError: If n_samples is nonpositive.
        """

    @abc.abstractmethod
    def size(self: "Dataset[T]") -> Optional[int]:
        """Returns the number of samples in this dataset, ie the epoch size.

        Returns None if not known or undefined.
        """
