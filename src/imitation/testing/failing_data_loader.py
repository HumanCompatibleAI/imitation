"""Broken Dataloader."""


class DataLoaderThatFailsOnNthIter:
    """A dummy DataLoader stops to yield after a number of calls to `__iter__`."""

    def __init__(self, dummy_yield_value: dict, no_yield_after_iter: int = 1):
        """Builds dummy data loader.

        Args:
            dummy_yield_value: The value to yield on each call.
            no_yield_after_iter: `__iter__` will raise `StopIteration` after
                this many calls.
        """
        self.iter_count = 0
        self.dummy_yield_value = dummy_yield_value
        self.no_yield_after_iter = no_yield_after_iter

    def __iter__(self):
        if self.iter_count < self.no_yield_after_iter:
            yield self.dummy_yield_value
        self.iter_count += 1
