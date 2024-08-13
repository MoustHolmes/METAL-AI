from torch.utils.data import Sampler
import random

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.grouped_indices = dataset.grouped_indices
        self.group_lengths = sorted(self.grouped_indices.keys())

    def __iter__(self):
        for length in self.group_lengths:
            indices = self.grouped_indices[length]
            random.shuffle(indices)
            # Yield full batches only
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size <= len(indices):
                    yield indices[i:i + self.batch_size]

    def __len__(self):
        # This is an approximation, as we drop the last incomplete batch in each group
        return sum(len(indices) // self.batch_size for indices in self.grouped_indices.values())