import torch
from torch.utils.data import Sampler

class OverlapBatchSampler(Sampler):
    """
    Batch sampler that yields batches with a fixed overlap fraction
    between consecutive batches.

    Example: batch_size=100, new_fraction=0.1
    => 90 samples overlap with previous batch, 10 are new.
    """
    def __init__(self, data_source, batch_size: int, new_fraction: float = 0.1, drop_last: bool = True):
        self.data_source = data_source
        self.n = len(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert 0.0 < new_fraction <= 1.0, "new_fraction must be in (0, 1]."
        # how many *new* samples per batch
        self.k_new = max(1, int(round(batch_size * new_fraction)))
        # stride of the sliding window
        self.step = self.k_new

    def __iter__(self):
        # new random permutation each epoch
        indices = torch.randperm(self.n).tolist()
        B = self.batch_size
        step = self.step

        pos = 0
        # sliding window: [pos : pos + B], then move by 'step'
        while pos + B <= self.n:
            yield indices[pos:pos + B]
            pos += step

        # Optional: handle leftover tail without guaranteed overlap ratio
        if (not self.drop_last) and (pos < self.n):
            # here we just take the last B indices to form a full batch
            if self.n >= B:
                yield indices[-B:]
            else:
                # dataset smaller than batch size; return smaller batch
                yield indices[pos:]

    def __len__(self):
        if self.n < self.batch_size:
            return 0 if self.drop_last else 1

        # count how many windows the while-loop will produce
        B = self.batch_size
        step = self.step

        cnt = 0
        pos = 0
        while pos + B <= self.n:
            cnt += 1
            pos += step

        if (not self.drop_last) and (pos < self.n):
            cnt += 1
        return cnt